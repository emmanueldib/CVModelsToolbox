"""This is a small helpee script for when you need to load your last.pt weights from HF back to disk. 
This can be useful for example if you switch Runpod instance mid-training, and need to get your last epoch's result back from the cloud on the new pod."""

from huggingface_hub import hf_hub_download
import argparse
from pathlib import Path
import shutil

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--repo", type=str, help="HF repo (YourUsername/YourRepo)")
    p.add_argument("--filename", type=str, help="The name of the downloaded file, e.g. last.pt")
    p.add_argument("--localdir", type=str, help="Where you want to save it, e.g. checkpoints/")
    p.add_argument("--outname", type=str, help="Name of the downloaded file")
    p.add_argument("--token", type=str, help="Your HF token")
    return p.parse_args()

def download_weights(args):
    cached = hf_hub_download(
        repo_id=args.repo,
        filename=args.filename,          # path inside the repo, e.g. "checkpoints/model.pt"
        token=args.token,               # only needed if the repo is private
    )
    dest=Path(args.localdir)/args.outname
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached,dest)
    return dest
    
if __name__=="__main__":
    args=parse_args()
    path=download_weights(args)
    print(f"Weights {args.filename} saved to {args.localdir}")