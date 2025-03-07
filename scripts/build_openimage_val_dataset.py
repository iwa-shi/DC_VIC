"""
Create a OpenImage Validation Dataset for Beta-selection
Save VQGAN code indices as well for off-line encoding
"""

import argparse
import os
import random
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from tqdm import tqdm

import ldm
from ldm.models.autoencoder import VQModelInterface


def get_vqgan_config(vqgan_type: str):
    if vqgan_type == "f8-n256":
        vq_config = dict(
            embed_dim=4,
            n_embed=256,
            monitor=None,
            ddconfig=dict(
                double_z=False,
                z_channels=4,
                resolution=256,
                in_channels=3,
                out_ch=3,
                ch=128,
                ch_mult=[1, 2, 2, 4],
                num_res_blocks=2,
                attn_resolutions=[32],
                dropout=0.0,
            ),
            lossconfig=dict(target="torch.nn.Identity"),
        )
    elif vqgan_type == "f8":
        vq_config = dict(
            embed_dim=4,
            n_embed=16384,
            monitor=None,
            ddconfig=dict(
                double_z=False,
                z_channels=4,
                resolution=256,
                in_channels=3,
                out_ch=3,
                ch=128,
                ch_mult=[1, 2, 2, 4],
                num_res_blocks=2,
                attn_resolutions=[32],
                dropout=0.0,
            ),
            lossconfig=dict(target="torch.nn.Identity"),
        )
    elif vqgan_type == "f16":
        vq_config = dict(
            embed_dim=8,
            n_embed=16384,
            monitor=None,
            ddconfig=dict(
                double_z=False,
                z_channels=8,
                resolution=256,
                in_channels=3,
                out_ch=3,
                ch=128,
                ch_mult=[1, 1, 2, 2, 4],
                num_res_blocks=2,
                attn_resolutions=[16],
                dropout=0.0,
            ),
            lossconfig=dict(target="torch.nn.Identity"),
        )
    else:
        raise ValueError()
    return vq_config

def build_vqgan(vq_config: dict, weight_path: str, device: str) -> VQModelInterface:
    vae = VQModelInterface(**vq_config)
    state_dict = torch.load(weight_path, map_location=device)["state_dict"]
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("loss.")}
    load_msg = vae.load_state_dict(state_dict)
    vae.quantize.sane_index_shape = True
    vae.eval()
    print(load_msg)
    return vae.to(device)

def read_img(img_path: str) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    return img

@torch.no_grad()
def vqgan_encode(vae: VQModelInterface, img_batch: Tensor) -> Tensor:
    z = vae.encode(img_batch)
    qz, _, (_, _, indices) = vae.quantize(z)
    return indices


def main(args):
    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    img_dir = os.path.join(args.openimage_root, 'validation')
    save_root = args.save_root
    vqgan_ckpt_dir = "../checkpoint/vq_compression/pretrained_vq_model"

    patch_size = 256
    min_short_length = 256
    num_images = args.num_img

    vqgan_type = args.vqgan_type  # e.g. 'f8-n256'
    assert vqgan_type in ["f8-n256", "f16", "f8"]

    img_path_list = glob(os.path.join(img_dir, "*.jpg"))
    img_path_list.sort()
    np.random.shuffle(img_path_list)
    print(len(img_path_list), "images")

    save_dir_name = {"f8-n256": "vq_f8_n256", "f16": "vq_f16", "f8": "vq_f8"}[vqgan_type]
    save_dir = os.path.join(save_root, f"{save_dir_name}/crop_{patch_size}_{num_images}_seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)
    print("Save to", save_dir)

    vq_config = get_vqgan_config(vqgan_type)
    weight_path = os.path.join(vqgan_ckpt_dir, f"vq-{vqgan_type}.ckpt")
    assert os.path.exists(weight_path)

    vae = build_vqgan(vq_config, weight_path, device=args.device)

    transform = T.Compose(
        (
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        )
    )

    cnt = 0

    qbar = tqdm(total=num_images, ncols=80)

    for img_path in img_path_list:
        img = read_img(img_path)
        w, h = img.size

        if min(w, h) < min_short_length:
            continue

        img_name = os.path.basename(img_path).replace('.jpg', '.png')

        top = random.randint(0, h - patch_size)
        left = random.randint(0, w - patch_size)
        img = img.crop((left, top, left + patch_size, top + patch_size))

        img_torch = transform(img.copy()).unsqueeze(0).to(args.device)

        indices = vqgan_encode(vae, img_torch)
        indices_np = indices.cpu().numpy()

        if vqgan_type == "f8-n256":
            assert indices_np.max() < 256 and indices_np.min() >= 0
            indices_np = indices_np.astype(
                np.uint8
            )  # NOTE: To save datasize. It's applicable only for VQ-f8-n256
        else:
            assert indices_np.max() < 16384 and indices_np.min() >= 0
            indices_np = indices_np.astype(np.uint16)

        save_path = os.path.join(save_dir, img_name)
        new_image = np.array(img, dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, new_image)
        np.save(save_path.replace('.png', '.npy'), indices_np[0])

        cnt += 1
        qbar.update(1)
        if cnt == num_images:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqgan_type", type=str, choices=["f8-n256", "f16", "f8"])
    parser.add_argument('--openimage_root', type=str)
    parser.add_argument('--save_root', type=str, default="./dataset")
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-n", "--num_img", type=int, default=2000)
    args = parser.parse_args()
    main(args)
