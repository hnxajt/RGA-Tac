# RGA-Tac: Retrieval-Graph-Augmented Tactile Description Generation

RGA-Tac builds on TVL for retrieval-graph-augmented tactile description generation. Given a visual image and a tactile image, it retrieves tactile attributes from training cases and uses fine-tuned TVL-LLaMA to generate a tactile description.

## Environment and resources

Use Python 3.10 and an NVIDIA GPU. The table lists the default resource paths.

| Resource | Path |
| --- | --- |
| SSVTP data | `data/TVL-revise/tvl_dataset/ssvtp/` |
| TVL source | `vendor/tvl/` |
| LLaMA-2-7B and tokenizer | `weights/llama-2/` |

```bash
git clone https://github.com/Max-Fu/tvl.git vendor/tvl
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e '.[tvl]'
```

Download the TVL-revise archive parts to `data/TVL-revise/`, then merge and extract the archive. [TVL-revise dataset](https://huggingface.co/datasets/yoorhim/TVL-revise)

```bash
cd data/TVL-revise
zip -s0 tvl_dataset_sharded.zip --out tvl_dataset.zip
unzip tvl_dataset.zip
cd ../..
```

The default tactile encoder is `small`. Save the two checkpoints for your selected backbone under `weights/`. [TVL weights on Hugging Face](https://huggingface.co/mlfu7/Touch-Vision-Language-Models)

| `--backbone` | TVL tactile encoder | TVL-LLaMA |
| --- | --- | --- |
| `tiny` | `tvl_enc_vittiny.pth` | `tvl_llama_vittiny.pth` |
| `small` | `tvl_enc_vits.pth` | `tvl_llama_vits.pth` |
| `base` | `tvl_enc_vitb.pth` | `tvl_llama_vitb.pth` |

Place the LLaMA-2-7B weights and tokenizer in the following directory layout.

```text
weights/llama-2/
  tokenizer.model
  llama-2-7b/
    params.json
    consolidated.00.pth
```

## Prepare data and train

Run these commands from the repository root.

```bash
rga-tac-prepare --dataset-root data/TVL-revise/tvl_dataset

rga-tac-train
```

Training fine-tunes the LLaMA norm, bias, and LoRA parameters. All three backbones share the same training settings. [Training configuration](configs/t/training/rga_tac_small_train.json)

| Setting | Default |
| --- | --- |
| LoRA rank | 16 |
| Epochs | 4 |
| Batch size / gradient accumulation | 8 / 4 |
| Base learning rate / scaled learning rate | 0.001 / 0.000125 |
| Weight decay | 0.05 |
| Warmup epochs | 0 |

Training saves the checkpoint with the lowest validation loss to `training/checkpoints/checkpoint-best.pth` under the run's output directory.

## Generate tactile descriptions

```bash
rga-tac-demo
```

The demo generates a description for the first SSVTP test sample by default.

When switching to `tiny` or `base`, use the same `--backbone` value for training and generation. The default output is `outputs/ssvtp-<backbone>/generated.json`.

## Related information

The data and initial models come from [TVL](https://tactile-vlm.github.io/) and its paper, *A Touch, Vision, and Language Dataset for Multimodal Alignment* (Fu et al., ICML 2024).
