# CV Models Toolbox

Personal library of computer vision models implemented in PyTorch, trained on ImageNet-1k streamed/downloaded as WebDataset shards from Hugging Face (`timm/imagenet-1k-wds`).

## Setup

    pip install -r requirements.txt
    hf auth login   # ImageNet is gated: accept the license on HF first

## Train

    # quick smoke test (streams 1 shard)
    python train.py --shards 1 --epochs 1 --workers 0 --ckpt-dir ./checkpoints

    # real run on a GPU box (downloads shards to local disk)
    python train.py --shards 128 --download --data-root /data/imagenet-wds \
                    --ckpt-dir /workspace/checkpoints

See `python train.py --help` for all options.

## Layout

- `models/` — architectures (currently VGG-11)
- `data.py` — WebDataset loaders (streaming or download-to-disk)
- `engine.py` — train/eval loops, atomic checkpointing
- `train.py` — entry point

## Notes

- Checkpoints save `model`, `opt`, `epoch`, `args`; training auto-resumes
  from `last.pt` if present in `--ckpt-dir`.
- Weights are stored on the HF Hub, not in this repo.
- Only download shards to disk if storage is plenty (whole ImageNet dataset is ~150GB). Otherwise, stream them from HF without saving. Download mode is enabled by passing ```--download```, otherwise defaults to streaming.