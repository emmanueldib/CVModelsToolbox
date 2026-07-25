import argparse
import torch
import torch.nn as nn
from pathlib import Path
from  hub import push_weights
from models import VGGA_model
from imagenet_data import make_imagenet_loaders_streaming, make_imagenet_loaders_download
from engine import one_train_epoch, evaluate, save_checkpoint

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", type=int, default=11,help="number of train shards to use (1-1024).")
    p.add_argument("--download", action="store_true", help="Downloads the file to cache (if not passed, streams from the internet without saving the files to disk).")
    p.add_argument("--data-root", type=str, default="/data/imagenet-wds", help="The folder in which the data is cached, when in download mode (irrelevant if --download is not passed)")
    p.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    p.add_argument("--lr", type=float, default=0.01, help="The learning rate.")
    p.add_argument("--momentum", type=float, default=0.9, help="The momentum.")
    p.add_argument("--workers", type=int, default=8, help="The number of workers.")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints", help="The directory where the latest and best states are saved. Default is checkpoints in current folder, for runpod pass /workspace/checkpoints .")
    p.add_argument("--stop-train", type=int, default=5, help="How many training loops without improvement to run before considering the training to be finished.")
    p.add_argument("--epochs", type=int, default=10, help="The maximum number of epochs to run. The loop will still stop earlier if stop-train triggers.")
    p.add_argument("--push-to-hub", action="store_true", help="Push best weights to HF after training completes (must specify HF path with --path-in-repo)")
    p.add_argument("--path-in-repo", type=str, help="Path in HF repo to store the weights at, if enabled.")

    args=p.parse_args()
    if args.push_to_hub and not args.path_in_repo:
         p.error("--push-to-hub requires --path-in-repo")

    return p.parse_args()

def main():
    steps_without_improvement=0
    args=parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device={device}, args={args}")


    make_loaders = make_imagenet_loaders_download if args.download else make_imagenet_loaders_streaming
    kwargs={"local_root": args.data_root} if args.download else {}
    train_loader, test_loader= make_loaders(
        n_train_shards=args.shards,
        batch_size=args.batch_size,
        num_workers=args.workers,
        **kwargs
    )

    model=VGGA_model(1000).to(device)
    loss_fn=nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    ckpt_dir=Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    last_path=Path(args.ckpt_dir)/"last.pt"
    best_path=Path(args.ckpt_dir)/"best.pt"

    if last_path.is_file() and best_path.is_file():
        last_ckpt=torch.load(last_path, map_location=device, weights_only=True)
        best_ckpt=torch.load(best_path, map_location=device, weights_only=True)

        last_epoch=last_ckpt["epoch"]
        last_acc, best_acc=last_ckpt["acc"], best_ckpt["acc"]
        model.load_state_dict(last_ckpt["model"])
        optimizer.load_state_dict(last_ckpt["opt"])
        scheduler.load_state_dict(last_ckpt["scheduler"])
        print(f"Previous training checkpoints located. Resuming training from epoch {last_epoch+1}. \nLast recorded accuracy :{last_acc}, best recorded accuracy : {best_acc}")

    else:
        best_acc=0.0
        last_epoch=0

    for epoch in range(last_epoch+1, args.epochs+1):
        tr_loss, tr_acc= one_train_epoch(model, loss_fn=loss_fn, optimizer=optimizer, train_loader=train_loader, device=device)
        te_loss, te_acc= evaluate(model, loss_fn=loss_fn, test_loader=test_loader, device=device)

        scheduler.step()
        state={
            "model":model.state_dict(),
            "opt":optimizer.state_dict(),
            "acc":te_acc,
            "epoch":epoch,
            "scheduler":scheduler.state_dict(),
            "args":vars(args)
        }
        save_checkpoint(state,last_path)

        print(f"Epoch {epoch} :\nTrain loss = {tr_loss:.4f}, train accuracy = {tr_acc:.4f}\nTest loss = {te_loss:.4f}, test accuracy = {te_acc:.4f}\nCurrent lr : {optimizer.param_groups[0]["lr"]}")

        if te_acc>best_acc:
            best_acc=te_acc
            steps_without_improvement=0
            save_checkpoint(state,best_path)
            
        else:
            steps_without_improvement+=1
        
        if steps_without_improvement>=args.stop_train:
            print(f"{args.stop_train} epochs have elapsed without improvement, ending training.")
            break
    if args.push_to_hub and best_path.is_file():
                    push_weights(best_path, args.path_in_repo,commit_message=f"Model weights pushed to hub (val acc={best_acc:.3f}, shards={args.shards})")
        


if __name__ == "__main__":
    main()


