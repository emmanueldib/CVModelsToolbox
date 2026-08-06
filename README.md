# CV Models Toolbox

Personal library of computer vision models implemented in PyTorch, ready for training on ImageNet-1k streamed/downloaded as WebDataset shards from Hugging Face (`timm/imagenet-1k-wds`).

## Setup
For non CUDA-devices :
    pip install -r requirements.txt

For CUDA devices :
    pip install -r requirements_cuda.txt

Then sign into HF (ImageNet is gated: accept the license on HF first)
    hf auth login   

## Train

    # quick smoke test (streams 1 shard)
    python train.py --shards 1 --epochs 1 --workers 0 --ckpt-dir ./checkpoints

    # real run on a GPU box (downloads shards to local disk)
    python train.py --shards 128 --download --data-root /data/imagenet-wds \
                    --ckpt-dir /workspace/checkpoints

See `python train.py --help` for all options.

## Layout

- `models/` — architectures
- `data.py` — WebDataset loaders (streaming or download-to-disk)
- `engine.py` — train/eval loops, atomic checkpointing
- `train.py` — entry point

## Models so far

The models/ folder currently contains pytorch implementations for VGG architectures, the four typical ResNets (with the "ResNet 1.5" convention, see code), and MobileNet. The list may grow.

## Notes

- Checkpoints save `model`, `opt`, `epoch`, `args`; training auto-resumes from `last.pt` if present in `--ckpt-dir`.
- This repo contains no weights, just the models and the train.py loop to produce your own.
- Only download shards to disk if storage is plenty (whole ImageNet dataset is ~150GB). Otherwise, stream them from HF without saving. Download mode is enabled by passing ```--download```, otherwise defaults to streaming.