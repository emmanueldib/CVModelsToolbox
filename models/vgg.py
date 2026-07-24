import torch
import torch.nn as nn

class VGGA_model(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    self.layers=nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=64,kernel_size=3,stride=1,padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Flatten(),
      nn.Linear(in_features=25088, out_features=4096),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(in_features=4096, out_features=4096),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(in_features=4096, out_features=num_classes)
    )
    self._initialize_weights()

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity="relu")
        if m.bias is not None:
          nn.init.constant_(m.bias,0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight,0,0.01)
        nn.init.constant_(m.bias,0)

  def forward(self, x):
    return self.layers(x)