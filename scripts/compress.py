import argparse
import json
import os
from glob import glob
from typing import Dict, List

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.models import build_comp_model
from src.utils import img_utils
from src.utils.codec_utils import load_byte_strings, save_byte_strings
from src.utils.logger import get_root_logger
from src.utils.options import BaseConfig
import src  # to register classes

torch.backends.cudnn.deterministic = True


class CustomConfig(BaseConfig):
    @classmethod
    def get_opt(cls) -> 'CustomConfig':
        arg_dict = cls.arg_parse()
        filename = arg_dict['config_path']
        cfg_dict, cfg_text, loaded_yamls = cls._file2dict_yaml(filename)
        arg_dict = cls._merge_a_into_b(arg_dict, cfg_dict)
        arg_dict['is_train'] = False
        return cls(arg_dict, cfg_text=cfg_text, filename=filename)

    @staticmethod
    def arg_parse() -> Dict:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config_path', type=str, help='path to .yaml')
        parser.add_argument('--model_path', type=str, help='path to model')
        parser.add_argument('--img_dir', type=str)
        parser.add_argument('--save_dir', type=str)
        parser.add_argument('-q', '--quality', type=int, required=True)
        parser.add_argument('--decompress', action='store_true')
        parser.add_argument('-d', '--device', type=str, default='cuda:0')
        args = parser.parse_args()
        out_dict = vars(args) # argparse.Namespace -> Dict
        return out_dict

class ImageDataset(Dataset):
    def __init__(self, img_path_list: List[str]) -> None:
        super().__init__()
        self.img_path_list = img_path_list
        self.img_path_list.sort()
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.img_path_list)

    def __getitem__(self, index: int) -> Dict:
        img_path = self.img_path_list[index]
        img_name = os.path.basename(img_path)
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return {'real_images':img, 'img_name': img_name}


def get_rate_summary(out_dict: Dict) -> Dict:
    N, _, H, W = out_dict['real_images'].size()
    num_pixel = H * W * N
    bpp = out_dict['bpp'].item()
    total_bit = bpp * num_pixel
    return {
        'num_pixel': num_pixel,
        'total_bit': total_bit,
        'bitrate': bpp,
    }


def main():
    opt = CustomConfig.get_opt()
    logger = get_root_logger()
    save_dir = opt.save_dir
    os.makedirs(save_dir, exist_ok=True)

    img_path_list = glob(os.path.join(opt.img_dir, '*.png'))
    img_path_list.sort()
    dataset = ImageDataset(img_path_list)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    compress_kwargs = {}
    compress_kwargs['quality_ind'] = opt.quality

    model = build_comp_model(opt).to(opt.device)
    model.load_learned_weight(ckpt_path=opt.model_path)
    model.codec_setup()

    rate_summary_list = []

    for data_dict in tqdm(dataloader, ncols=80):
        img_name = data_dict['img_name'][0]
        img = data_dict['real_images']
        _, _, H, W = img.shape

        out_dict = model.compress(img, **compress_kwargs)

        string_list = out_dict['string_list']
        bin_path = os.path.join(save_dir, img_name.replace('.png', '.bin'))
        save_byte_strings(bin_path, string_list)
        actual_byte = os.path.getsize(bin_path)

        data_dict = {
            'img_name': img_name,
            'header_bit': len(string_list[0]) * 8,
            'z_bit': len(string_list[1]) * 8,
            'y_bit': len(string_list[2]) * 8,
            'real_bit': actual_byte * 8,
            'real_bpp': actual_byte * 8 / H / W,
            'pred_z_bit': out_dict['pred_z_bit'],
            'pred_y_bit': out_dict['pred_y_bit'],
            'pred_bit': out_dict['pred_z_bit'] + out_dict['pred_y_bit'],
            'pred_bpp': out_dict['pred_z_bpp'] + out_dict['pred_y_bpp'],
            'num_pixel': H * W,
        }
        rate_summary_list.append(data_dict.copy())

        if opt.decompress:
            loaded_string_list = load_byte_strings(bin_path)
            fake_img, z_hat, y_hat = model.decompress(loaded_string_list)
            recon_path = os.path.join(save_dir, img_name)
            img_utils.imwrite(recon_path, fake_img)
            # print('y_hat allclose:', torch.allclose(y_hat, out_dict['y_hat'])) # for debug should be True
            # print('z_hat allclose:', torch.allclose(z_hat, out_dict['z_hat']))

    # save bit-rate summary
    df = pd.json_normalize(rate_summary_list)
    df.to_csv(os.path.join(save_dir, '_bitrates.csv'))
    with open(os.path.join(save_dir, '_avg_bitrate.json'), 'w') as f:
        json.dump({'avg_bpp': df['real_bpp'].mean()}, f)

if __name__ == '__main__':
    main()
