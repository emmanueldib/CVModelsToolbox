from torchvision import transforms
from pathlib import Path
import webdataset as wds
from huggingface_hub import get_token, snapshot_download

IMAGENET_MEAN=[0.485, 0.456, 0.406]
IMAGENET_STD=[0.229, 0.224, 0.225]

def _identity(x):
    return x

def _build_loaders(train_urls, val_urls, n_samples, batch_size=128, num_workers=8, shuffle_buffer=1000):
    
    

    train_transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN,
                            std=IMAGENET_STD),
    ])

    train_dataset = (
    wds.WebDataset(train_urls, shardshuffle=1000, nodesplitter=wds.split_by_node, handler=wds.warn_and_continue)
    .shuffle(shuffle_buffer)
    .decode("pil", handler=wds.warn_and_continue)
    .to_tuple("jpg","cls", handler=wds.warn_and_continue)
    .map_tuple(train_transform, _identity)
    .batched(batch_size)
    )

    train_loader=wds.WebLoader(
    train_dataset,
    batch_size=None,
    num_workers=num_workers,
    pin_memory=True,
    )

    val_transform=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


    test_dataset = (
        wds.WebDataset(val_urls, shardshuffle=False, nodesplitter=wds.split_by_node, handler=wds.warn_and_continue)
        .decode("pil", handler=wds.warn_and_continue)
        .to_tuple("jpg","cls", handler=wds.warn_and_continue)
        .map_tuple(val_transform, _identity)
        .batched(batch_size)
    )



    test_loader=wds.WebLoader(
        test_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
    )

    n_batches=n_samples//batch_size
    train_loader=train_loader.with_epoch(n_batches)

    return train_loader, test_loader


def make_imagenet_loaders_streaming(n_train_shards=1024, **kw):

    hf_token = get_token()
    if hf_token is None:
        raise RuntimeError("No huggingface token found. Login with hf auth login, then retry.")

    last=n_train_shards-1
    n_samples=1281167 if n_train_shards==1024 else n_train_shards*1250

    


    base = "https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main" 
    train_urls = f"pipe:curl --retry 3 --retry-connrefused -s -f -L {base}/imagenet1k-train-{{0000..{last:04d}}}.tar -H 'Authorization:Bearer {hf_token}'" 
    val_urls = f"pipe:curl --retry 3 --retry-connrefused -s -f -L {base}/imagenet1k-validation-{{00..63}}.tar -H 'Authorization:Bearer {hf_token}'"

    return _build_loaders(train_urls=train_urls, val_urls=val_urls, n_samples=n_samples, **kw)

def make_imagenet_loaders_download(n_train_shards,local_root="/data/imagenet-wds",**kw):

    last=n_train_shards-1
    n_samples=1281167 if n_train_shards==1024 else n_train_shards*1250

    train_urls=f"{local_root}/imagenet1k-train-{{0000..{last:04d}}}.tar"
    val_urls   = f"{local_root}/imagenet1k-validation-{{00..63}}.tar"

    train_files=[f"imagenet1k-train-{i:04d}.tar" for i in range(n_train_shards)]
    val_files=[f"imagenet1k-validation-{i:02d}.tar" for i in range(64)]

    snapshot_download(
    "timm/imagenet-1k-wds", repo_type="dataset",
    local_dir=local_root,
    allow_patterns=train_files+val_files,     
    )
    

    return _build_loaders(train_urls=train_urls, val_urls=val_urls,n_samples=n_samples, **kw)
    

