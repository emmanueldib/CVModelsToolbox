import os
import torch
from tqdm.auto import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from imagenet_data import VAL_TRANSFORM
from torchvision.models import ResNet50_Weights
import numpy as np

CATEGORIES=ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

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

def predict(model, img_path, device, topk=10, categories=CATEGORIES, transform=VAL_TRANSFORM):
  """img_tensor must be of shape (1, 3, height, width), output of Image.open(img_path)"""
  image=Image.open(img_path).convert("RGB")
  x=transform(image).unsqueeze(0).to(device)
  model.eval()
  with torch.inference_mode():
    preds=torch.softmax(model(x), dim=1)
  probs, idx=torch.topk(preds, k=topk)
  print(probs, idx)
  return probs.squeeze(), [categories[i] for i in idx.squeeze()]

def prediction_graph(model, img_paths_batch, device, topk=10, transform=VAL_TRANSFORM):
  probs_list, cats_list = [], []
  for i in range(len(img_paths_batch)):
    probs, cats=predict(model, img_paths_batch[i], device, topk, CATEGORIES, transform)
    probs_list.append(probs)
    cats_list.append(cats)
  fig, axes= plt.subplots(figsize=(10, 4.5*len(img_paths_batch)), nrows=len(img_paths_batch), ncols=2, squeeze=False)
  for i in range (len(img_paths_batch)):
    img=np.array(Image.open(img_paths_batch[i]).convert("RGB"))
    print(cats_list[i])
    axes[i,0].imshow(img)
    axes[i,1].barh(cats_list[i], probs_list[i])
    axes[i,1].invert_yaxis()
    plt.tight_layout()

