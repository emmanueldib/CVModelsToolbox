import os
import torch
from tqdm.auto import tqdm

def one_train_epoch(model, loss_fn, optimizer, train_loader, device):

  model.train()

  tr_loss, tr_correct, tr_seen=0.0,0,0

  for images, labels in tqdm(train_loader, desc="Training..."):
    images=images.to(device)
    labels=labels.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
      logits=model(images)
      train_loss=loss_fn(logits,labels)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    tr_loss+=train_loss.item()*labels.size(0)
    tr_correct+=(logits.argmax(dim=1)==labels).sum().item()
    tr_seen+=labels.size(0)
  

  tr_loss/=tr_seen
  tr_acc=tr_correct/tr_seen
  

  return tr_loss, tr_acc

def evaluate(model, loss_fn, test_loader, device):

  model.eval()
  with torch.inference_mode():
    te_loss, te_correct, te_seen= 0.0, 0, 0
    for images, labels in tqdm(test_loader, desc="Testing..."):
      images=images.to(device)
      labels=labels.to(device)
      logits=model(images)

      te_loss+=loss_fn(logits,labels).item() * labels.size(0)
      te_correct+=(logits.argmax(dim=1)==labels).sum().item()
      te_seen+=labels.size(0)

  te_loss/=te_seen
  te_acc=te_correct/te_seen

  return te_loss, te_acc



def save_checkpoint(state, filename):
  """Call this as : save_checkpoint({"model": model.state_dict(),
                 "opt": optimizer.state_dict(),
                 "epoch": epoch}, "checkpoints/last.pt")"""
  tmp=str(filename)+".tmp"
  torch.save(state,tmp)
  os.replace(tmp,filename)
