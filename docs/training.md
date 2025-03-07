# Training

## Dataset Preparation

#### Training dataset

For training, we used the [OpenImage](https://storage.googleapis.com/openimages/web/index.html) dataset.
Specifically, we used subsets `train_0` through `train_9` (~1.1M images), which can be downloaded from [here](https://github.com/cvdfoundation/open-images-dataset?tab=readme-ov-file#download-images-with-bounding-boxes-annotations).
Place each sub-directory (`train_X`) in a main directory. Additionally, we used the `validation` subset for the beta-selection stage.
```
openimage/
├── train_0/
│   ├── XXXXXX.jpg
│   ├── ...
│   └── YYYYYY.jpg
├── train_1/
├── ...
├── train_9/
└── validation/
```


> [!NOTE]
> If you cannot download all 10 sub-directories of OpenImage, you can download some of them and modify the `subset_list` key in the config accordingly.
> For example, change `subset_list: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` to `subset_list: [0, 1]` if you only have sub-directories `train_0` and `train_1`.

#### Validation dataset

For validation, we used the [Kodak](https://r0k.us/graphics/kodak/) dataset. The dataset directory should be structured as follows:
```
kodak/
├── kodim01.png
├── kodim02.png
├── ...
└── kodim24.png
```


After preparing the datasets, specify their paths in the `root_dir` key in [config/_base_/dataset/openimage_kodak.yaml](../config/_base_/dataset/train_openimage_eval_kodak.yaml).


## Download pre-trained VQGAN checkpoint

The pre-trained VQGAN checkpoint can be found in the [official taming-transformers repository](https://github.com/CompVis/taming-transformers?tab=readme-ov-file#overview-of-pretrained-models). We use the **"VQGAN OpenImages (f=8), 256"** checkpoint as our default setting.

Download the checkpoint and specify the path in the `subnet.vq_model.ckpt_path` key within [config/_base_/model/hyperprior_charm_dual_cond_vic_model_vq_f8_n256.yaml](../config/_base_/model/hyperprior_charm_dual_cond_vic_model_vq_f8_n256.yaml).

Example:
```yaml
subnet:
  vq_model:
      ckpt_path: ./checkpoint/pretrained_vq_model/vq-f8-n256.ckpt
```

## Stage 1: Dual-Conditioned Pre-training (Sec3.C.1 in the paper)

The pre-training consists of 3 steps. For each step, you can run the training with the following command:
```
poetry run python ./scripts/train.py {CONFIG_PATH} -d {DEVICE}
```
Additional options:
- `-wb, --use_wandb`: Enable Weights & Biases logging. See `init_wandb()` in [`src/trainer/base_trainer.py`](../src/trainer/base_trainer.py)
- `--debug`: Run training in debug mode (more detailed logs will be shown).
- See [`src/utils/options.py`](../src/utils/options.py) for more options.

The config files for each step are [`exp1_stage1_1.yaml`](../config/exp1_stage1_1.yaml), [`exp1_stage1_2.yaml`](../config/exp1_stage1_2.yaml), and [`exp1_stage1_3.yaml`](../config/exp1_stage1_3.yaml).
Run the following three commands sequentially for the 3 steps:
```
# Step 1: High-bitrate warmup without dual-condition (500K iterations)
poetry run python ./scripts/train.py ./config/exp1_stage1_1.yaml -d {DEVICE}

# Step 2: without GAN training (500K iterations)
poetry run python ./scripts/train.py ./config/exp1_stage1_2.yaml -d {DEVICE}

# Step 3: GAN training (500K iterations)
poetry run python ./scripts/train.py ./config/exp1_stage1_3.yaml -d {DEVICE}
```

## Stage 2: Beta-selection (Sec3.C.2 in the paper)

After completing Stage 1 training, we select appropriate `beta_rate` and `beta_vq` pairs for each target bitrate using the following procedures:

#### Step 1: Create validation dataset
```
poetry run python scripts/build_openimage_val_dataset.py --vqgan_type f8-n256 \
--openimage_root ./datasets/openimage \
--save_root ./datasets/openimage_validation_vq \
--seed 0 \
--num_img 2000
```

#### Step 2: Binary search
Find the optimal `beta_rate` for each `beta_vq` so that the average bpp on the validation dataset matches the target bitrates.
```
poetry run python scripts/binary_rate_search.py ./config/exp1_stage1_3.yaml \
--model_path ./checkpoint/exp1_stage1_3/model/comp_model_iter500K.pth.tar \
--save_dir ./bin_search/exp1_stage1_3 \
--dataset_root ./datasets/openimage_validation_vq/vq_f8_n256/crop_256_2000_seed_0 \
--batch_size 16 \
--max_beta_rate 3.0 \
--beta_vq 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.25 3.5 \
--target_rate 0.05 0.075 0.1 0.125 0.15
```

#### Step 3: Evaluation and selection
```
poetry run python scripts/beta_selection.py ./config/exp1_stage1_3.yaml \
--model_path ./checkpoint/exp1_stage1_3/model/comp_model_iter500K.pth.tar \
--save_dir ./beta_selection/exp1_stage1_3 \
--search_dir ./bin_search/exp1_stage1_3 \
--dataset_root ./datasets/openimage_validation_vq/vq_f8_n256/crop_256_2000_seed_0 \
--batch_size 8 \
--beta_vq 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.25 3.5 \
--target_rate 0.05 0.075 0.1 0.125 0.15
```
Selected `beta_vq, beta_rate` pairs will be stored in `./bin_search/exp1_stage1_3/beta_selection_results.csv`.

Example output:
```
target_rate,selected_beta_vq,selected_beta_rate
0.05,3.0,2.297
0.075,2.25,1.511
0.1,2.0,1.125
0.125,1.5,0.621
0.15,1.0,0.164
```

> [!NOTE]
> You might want to test the above procedures using a smaller dataset or fewer search candidates to ensure everything works properly before running the full-scale search.

## Stage 3: Selected-beta Fine-tuning (Sec3.C.3 in the paper)
Set the selected `beta_rate` and `beta_vq` values in the `selected_beta_rate` and `selected_beta_vq` keys in the config file (`exp1_stage3.yaml`) as shown below:
```yaml
model:
  enc_vq_input: onehot_indices
  use_selected_beta_pairs: true
  selected_beta_rate: [
    2.29, # 0.050 bpp
    1.51, # 0.075 bpp
    1.12, # 0.100 bpp
    0.62, # 0.125 bpp
    0.16, # 0.150 bpp
  ]
  selected_beta_vq: [
    3.00, # 0.050 bpp
    2.25, # 0.075 bpp
    2.00, # 0.100 bpp
    1.50, # 0.125 bpp
    1.00, # 0.150 bpp
  ]
```

Then, start training (500K iterations):
```
poetry run python scripts/train.py ./config/exp1_stage3.yaml -d {DEVICE}
```

## Issues
If you have any questions or encounterd any issues, please feel free to open issue.