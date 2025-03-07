import argparse
import logging
import os
from datetime import datetime
from glob import glob
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

import src
from src.models import build_comp_model
from src.utils.logger import get_root_logger
from src.utils.options import BaseConfig

MEMO_DICT = {}
MAX_RUN_CNT = 10


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
        parser.add_argument("--save_dir", type=str)
        parser.add_argument("--dataset_root", type=str)
        parser.add_argument('--beta_vq', type=float, nargs='+')
        parser.add_argument('--target_rate', type=float, nargs='+')
        parser.add_argument('--max_beta_rate', type=float)
        parser.add_argument('--error_delta', type=float, default=0.001)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("-d", "--device", type=str, default="cuda:0")
        args = parser.parse_args()
        out_dict = vars(args)  # argparse.Namespace -> Dict
        return out_dict


class ImageVqTokenDataset(Dataset):
    def __init__(self, root_dir: str) -> None:
        super().__init__()
        self.img_list = glob(os.path.join(root_dir, '*.png'))
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, index: int) -> Dict:
        img_path = self.img_list[index]
        img = Image.open(img_path).convert('RGB')
        vq_indices = np.load(img_path.replace('.png', '.npy'))

        # To tensor
        img = self.transform(img)
        vq_indices = vq_indices.astype(np.int32)
        vq_indices = torch.from_numpy(vq_indices).long()

        return {'real_images':img, 'vq_indices': vq_indices}

def build_pretrianed_model(opt: CustomConfig):
    model = build_comp_model(opt).to(opt.device)
    model.load_learned_weight(ckpt_path=opt.model_path)
    return model

def build_dataloader(opt: CustomConfig):
    logger = get_root_logger()
    dataset_root = opt.dataset_root
    assert os.path.exists(dataset_root), f'dataset_root "{dataset_root}" does not exist.'

    logger.info(f'dataset_root: {dataset_root}')
    dataset = ImageVqTokenDataset(dataset_root)
    batch_size = opt.batch_size
    num_workers = min(8, batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=num_workers)
    return dataloader

@torch.no_grad()
def run_one_search(model, dataloader, beta_rate: float, beta_vq: float) -> float:
    bpp_list = []

    for data_dict in tqdm(dataloader, ncols=80, leave=False):
        N, _, H, W = data_dict['real_images'].shape
        processed_inputs = model.data_preprocess(
            vq_indices=data_dict['vq_indices'],
            real_images=data_dict['real_images'],
            beta_rate=beta_rate,
            beta_vq=beta_vq,
            is_train=False,
        )
        real_images = processed_inputs['real_images']
        vq_indices = processed_inputs['vq_indices']

        gt_vq_latent, gt_vq_indices = model.vq_encode(
            real_images, vq_indices
        )
        y = model.comp_encode(
            real_images=real_images,
            gt_vq_latent=gt_vq_latent,
            gt_vq_indices=gt_vq_indices,
            enc_kwargs=dict(beta_1=beta_rate, beta_2=beta_vq),
        )
        entropy_dict = model.estimate_entropy(y, is_train=False)
        rate_summary_dict = model.get_rate_summary_dict(entropy_dict, num_pixel=N*H*W)
        bpp_list.append(rate_summary_dict['bpp'].item())

    return np.mean(bpp_list)

def run(opt, model, dataloader, target_rate, beta_vq, logger):
    data_list = []
    beta_rate_min = 0.0
    beta_rate_max = opt.max_beta_rate
    error_delta = opt.error_delta
    run_cnt = 0

    while True:
        run_cnt += 1
        beta_rate = (beta_rate_min + beta_rate_max) / 2.
        beta_rate = round(beta_rate, 3)

        memo_key = memo_dict_key(beta_vq, beta_rate)

        logger.info(f'run_cnt {run_cnt:2} | min: {beta_rate_min}, max: {beta_rate_max}, beta_rate: {beta_rate}')

        if memo_key in MEMO_DICT:
            logger.info(f'run_cnt {run_cnt:2} |   beta_rate: {beta_rate} is already searched')
            avg_bpp = MEMO_DICT[memo_key]
        else:
            logger.info(f'run_cnt {run_cnt:2} |   beta_rate: {beta_rate} is not searched yet. Start running...')
            avg_bpp = run_one_search(model, dataloader, beta_rate, beta_vq)
            MEMO_DICT[memo_key] = avg_bpp

        diff = abs(avg_bpp - target_rate)
        data_list.append({
            'run_cnt': run_cnt,
            'beta_vq': beta_vq,
            'beta_rate': beta_rate,
            'avg_bpp': avg_bpp,
            'diff': diff,
        })

        logger.info(f'run_cnt {run_cnt:2} |   avg_bpp: {avg_bpp:.5f}, diff: {diff:.5f}')

        if abs(avg_bpp - target_rate) <= error_delta:
            break
        elif avg_bpp > target_rate: # beta_rate is too small
            beta_rate_min = beta_rate
        else: # beta_rate is too large
            beta_rate_max = beta_rate

        if run_cnt >= MAX_RUN_CNT:
            logger.warning(f'reached max run count: {MAX_RUN_CNT}')
            break

    logger.info(f'binary search finished')

    df = pd.json_normalize(data_list)
    df = df.sort_values('diff').reset_index(drop=True)
    return df

def memo_dict_key(beta_vq: float, beta_rate: float) -> str:
    return f'{beta_vq:.4f}-{beta_rate:.4f}'.replace('.', '_')

def main() -> None:
    opt = CustomConfig.get_opt()
    beta_vq = opt.beta_vq
    target_rate = opt.target_rate

    search_dir = opt.save_dir
    os.makedirs(search_dir, exist_ok=True)

    timestamp = datetime.now().isoformat(timespec='seconds')
    log_file = os.path.join(search_dir,f'run_{timestamp}.log')
    logger = get_root_logger(log_level=logging.INFO, log_file=log_file)

    model = build_pretrianed_model(opt)
    dataloader = build_dataloader(opt)

    for i, (beta_vq, target_rate) in enumerate(product(opt.beta_vq, opt.target_rate)):
        logger.info(f'**** {i+1}/{len(opt.beta_vq) * len(opt.target_rate)} ****')
        logger.info(f'beta_vq: {beta_vq}, target_rate: {target_rate}')

        beta_vq_str = f'{beta_vq:.2f}'
        target_rate_str = f'{target_rate:.3f}'
        csv_path = os.path.join(search_dir, f'result_beta_vq_{beta_vq_str}_target_rate_{target_rate_str}.csv')
        df = run(
            opt, model, dataloader, target_rate, beta_vq, logger
        )
        df.to_csv(csv_path)




if __name__ == "__main__":
    main()

