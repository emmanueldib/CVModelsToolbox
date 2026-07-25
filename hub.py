from huggingface_hub import HfApi, hf_hub_download
import torch

def push_weights(local_path, path_in_repo, repo_id, commit_message="Updated weights"):
    """Upload a checkpoint to the HF hub"""
    HfApi().upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message
    )

def load_weights(model,path_in_repo,repo_id, device):
    """Download and load weights into a model"""
    path=hf_hub_download(repo_id, path_in_repo)
    ckpt=torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    return ckpt