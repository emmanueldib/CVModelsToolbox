import torch
import torch.nn as nn

def Conv3(ins, out, stride):
    return nn.Conv2d(kernel_size=3, in_channels=ins, out_channels=out, stride=stride, padding=1, bias=False)

def Conv1(ins, out, stride=1):
    return nn.Conv2d(kernel_size=1, in_channels=ins, out_channels=out, stride=stride, bias=False)

def Conv3dw(channels,stride=1):
    return nn.Conv2d(kernel_size=3, in_channels=channels, out_channels=channels, stride=stride, groups=channels,padding=1, bias=False)

class Block(nn.Module):
    def __init__(self,ins,out,t,stride):
        super().__init__()
        self.hidden=ins*t
        self.ins=ins
        self.out=out
        self.stride=stride
        layers=[]
        if t!=1:
            layers+=[Conv1(ins,self.hidden,1),nn.BatchNorm2d(self.hidden),nn.ReLU6(inplace=True)]

        layers+=[
            Conv3dw(self.hidden,stride),
            nn.BatchNorm2d(self.hidden),
            nn.ReLU6(inplace=True),
            Conv1(self.hidden,out,1),
            nn.BatchNorm2d(out)
        ]

        self.Layers=nn.Sequential(*layers)

    def forward(self,x):
        output=self.Layers(x)
        return output+x if (self.ins==self.out and self.stride==1) else output

class MobileNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.stem=nn.Sequential(
            Conv3(3,32,2),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        ts=[1,6,6,6,6,6,6]
        channels=[16,24,32,64,96,160,320]
        strides=[1,2,2,2,1,2,1]
        n_blocks=[1,2,3,4,3,3,1]
        layers=[]
        ins=32
        for stage in range(len(ts)):
            for block in range(n_blocks[stage]):
                s=strides[stage] if block==0 else 1
                layers+=[Block(ins,channels[stage],ts[stage],s)]
                ins=channels[stage]

        self.convlayers=nn.Sequential(*layers)

        self.head=nn.Sequential(
            Conv1(ins,1280,1), nn.BatchNorm2d(1280), nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280,num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)

    def forward(self, x):
        output=self.stem(x)
        output=self.convlayers(output)
        output=self.head(output)
        return(output)
        
        