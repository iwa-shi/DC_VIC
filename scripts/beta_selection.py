import argparse
import json
import logging
import os
import shutil
from datetime import datetime
from glob import glob
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from calc_metrics import FIDMetric, LPIPSMetric, PSNRMetric

import src
from src.models import build_comp_model
from src.utils import img_utils
from src.utils.logger import get_root_logger
from src.utils.options import BaseConfig

SEARCH_ERROR_THRESHOLD = 0.001


class CustomConfig(BaseConfig):
    @classmethod
    def get_opt(cls) -> "CustomConfig":
        arg_dict = cls.arg_parse()
        filename = arg_dict["config_path"]
        cfg_dict, cfg_text, loaded_yamls = cls._file2dict_yaml(filename)
        cfg_dict["loaded_yamls"] = loaded_yamls
        arg_dict = cls._merge_a_into_b(arg_dict, cfg_dict)
        return cls(arg_dict, cfg_text=cfg_text, filename=filename)

    @staticmethod
    def arg_parse() -> dict:
        parser = argparse.ArgumentParser()
        parser.add_argument("config_path", type=str)
        parser.add_argument("--model_path", type=str)
        parser.add_argument("--search_dir", type=str)
        parser.add_argument("--save_dir", type=str)
        parser.add_argument("--dataset_root", type=str)
        parser.add_argument("--beta_vq", type=float, nargs="+")
        parser.add_argument("--target_rate", type=float, nargs="+")
        parser.add_argument(
            "--alpha",
            type=float,
            default=2.0,
            help="score is calculated as alpha * PSNR - FID (see Eq.13 in the paper). Default: 2.0",
        )
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--keep_recon", action="store_true")
        parser.add_argument("-d", "--device", type=str, default="cuda:0")
        args = parser.parse_args()
        out_dict = vars(args)  # argparse.Namespace -> Dict
        return out_dict


class ImageVqTokenDataset(Dataset):
    def __init__(self, root_dir: str) -> None:
        super().__init__()
        self.img_list = glob(os.path.join(root_dir, "*.png"))
        self.img_list.sort()
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, index: int) -> Dict:
        img_path = self.img_list[index]
        img = Image.open(img_path).convert("RGB")
        vq_indices = np.load(img_path.replace(".png", ".npy"))

        # To tensor
        img = self.transform(img)
        vq_indices = vq_indices.astype(np.int32)
        vq_indices = torch.from_numpy(vq_indices).long()

        return {
            "real_images": img,
            "vq_indices": vq_indices,
            "img_name": os.path.basename(img_path),
        }


def calc_batch_bpp(
    y_likelihood: Tensor, z_likelihood: Tensor, num_pixel: int
) -> Tensor:
    bit_y = -torch.log(y_likelihood) / np.log(2)  # [N, C, H, W]
    bit_z = -torch.log(z_likelihood) / np.log(2)  # [N, C, H, W]
    bit_y = torch.sum(bit_y, dim=tuple(range(1, bit_y.ndim)))  # [N]
    bit_z = torch.sum(bit_z, dim=tuple(range(1, bit_z.ndim)))  # [N]
    bpp = (bit_y + bit_z) / num_pixel  # [N]
    return bpp


def get_rate_summary(img_name: str, bpp: Tensor, num_pixel: int) -> Dict:
    assert bpp.numel() == 1
    bpp_float = bpp.item()
    total_bit = bpp_float * num_pixel
    return {
        "img_name": img_name.split(".")[0],
        "num_pixel": num_pixel,
        "total_bit": total_bit,
        "bitrate": bpp_float,
    }


def save_reconstructions(model, dataloader, save_dir, beta_vq, beta_rate):
    rate_summary_list = []

    for data_dict in tqdm(dataloader, ncols=80):
        img_torch = data_dict["real_images"]
        vq_indices = data_dict["vq_indices"]
        _bs = data_dict["real_images"].size(0)
        with torch.no_grad():
            out_dict = model.run_model(
                real_images=img_torch,
                vq_indices=vq_indices,
                beta_vq=beta_vq,
                beta_rate=beta_rate,
                is_train=False,
            )

        H, W = img_torch.size()[-2:]
        batch_bpp = calc_batch_bpp(
            out_dict["y_likelihood"], out_dict["z_likelihood"], num_pixel=H * W
        )

        for i in range(_bs):
            img_name = data_dict["img_name"][i]
            save_path = os.path.join(save_dir, img_name)
            img_utils.imwrite(save_path, out_dict["fake_images"][i])
            rate_summary = get_rate_summary(img_name, batch_bpp[i], num_pixel=H * W)
            rate_summary_list.append(rate_summary.copy())

    df = pd.json_normalize(rate_summary_list)
    df.to_csv(os.path.join(save_dir, "_rate_summary.csv"))

    avg_bpp = df["bitrate"].mean()
    with open(os.path.join(save_dir, "_avg_bitrate.json"), "w") as f:
        json.dump({"avg_bpp": avg_bpp}, f)

    return avg_bpp


def main(opt):
    os.makedirs(opt.save_dir, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    log_file = os.path.join(opt.save_dir, f"run_{timestamp}.log")
    logger = get_root_logger(log_level=logging.INFO, log_file=log_file)

    ######################## Build Dataset ########################
    dataset_root = opt.dataset_root
    assert os.path.exists(
        dataset_root
    ), f'dataset_root "{dataset_root}" does not exist.'
    logger.info(f"dataset_root: {dataset_root}")

    dataset = ImageVqTokenDataset(dataset_root)
    batch_size = opt.batch_size
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=batch_size,
    )

    ######################## Build Model ########################
    model = build_comp_model(opt).to(opt.device)
    model.load_learned_weight(ckpt_path=opt.model_path)
    model.eval()

    ######################## Run ########################
    psnr_func = PSNRMetric()
    fid_func = FIDMetric(opt.device)

    selection_results = []

    for target_rate in opt.target_rate:
        data_list = []
        save_dir = os.path.join(opt.save_dir, f"target_rate_{target_rate}")
        os.makedirs(save_dir, exist_ok=True)

        for beta_vq in opt.beta_vq:
            beta_vq_str = f"{beta_vq:.2f}"
            target_rate_str = f"{target_rate:.3f}"
            bin_search_csv = os.path.join(
                opt.search_dir,
                f"result_beta_vq_{beta_vq_str}_target_rate_{target_rate_str}.csv",
            )
            bin_search_df = pd.read_csv(bin_search_csv).sort_values(by="diff")
            search_result = bin_search_df.iloc[0]
            if search_result["diff"] > SEARCH_ERROR_THRESHOLD:
                logger.warning(
                    f'bpp difference is larger than threshold: {search_result["diff"]} > {SEARCH_ERROR_THRESHOLD}. Skip.'
                )
                continue

            beta_rate = search_result["beta_rate"]
            recon_dir = os.path.join(save_dir, f"beta_vq_{beta_vq_str}")
            os.makedirs(recon_dir, exist_ok=True)
            avg_bpp = save_reconstructions(
                model, dataloader, recon_dir, beta_vq, beta_rate
            )

            fake_images = sorted(glob(os.path.join(recon_dir, "*.png")))
            real_images = sorted(glob(os.path.join(dataset_root, "*.png")))

            psnr_val = psnr_func.calc_metric(real_images, fake_images)
            fid_val = fid_func.calc_metric(real_images, fake_images)

            data_list.append({
                "beta_vq": beta_vq,
                "beta_rate": beta_rate,
                "bpp": avg_bpp,
                "psnr": psnr_val,
                "fid": fid_val,
                "score": opt.alpha * psnr_val - fid_val,
            })

            if not opt.keep_recon:
                shutil.rmtree(recon_dir)

        result_df = pd.json_normalize(data_list)
        result_df = result_df.sort_values(by="score", ascending=False)
        result_df.to_csv(os.path.join(save_dir, "result.csv"))
        best_result = result_df.iloc[0]
        logger.info(
            f'target_rate: {target_rate}, selected beta_vq: {best_result["beta_vq"]}, selected beta_rate: {best_result["beta_rate"]}'
        )
        selection_results.append({
            "target_rate": target_rate,
            "selected_beta_vq": best_result["beta_vq"],
            "selected_beta_rate": best_result["beta_rate"],
        })

    csv_path = os.path.join(opt.save_dir, "beta_selection_results.csv")
    pd.json_normalize(selection_results).to_csv(csv_path, index=False)


if __name__ == "__main__":
    opt = CustomConfig.get_opt()
    main(opt)
