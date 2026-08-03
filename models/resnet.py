import torch
import torch.nn as nn

def Conv7(ins, out, stride=1):
    return nn.Conv2d(kernel_size=7, in_channels=ins, out_channels=out, stride=stride, padding=3, bias=False)

def Conv3(ins, out, stride):
    return nn.Conv2d(kernel_size=3, in_channels=ins, out_channels=out, stride=stride, padding=1, bias=False)

def Conv1(ins, out, stride=1):
    return nn.Conv2d(kernel_size=1, in_channels=ins, out_channels=out, stride=stride, bias=False)


class BasicBlock(nn.Module):

    expansion=1

    def __init__(self, ins, out, stride):
        super().__init__()
        self.stride=stride
        self.ins=ins
        self.out=out

        self.cn1=Conv3(ins,out,stride)
        self.bn1=nn.BatchNorm2d(out)
        self.cn2=Conv3(out,out,1)
        self.bn2=nn.BatchNorm2d(out)
        self.relu=nn.ReLU(inplace=True)

        if stride!=1 or ins!=out:
            self.downsample=nn.Sequential(Conv1(self.ins,self.out,self.stride),nn.BatchNorm2d(self.out))
        else:
            self.downsample=None

    def forward(self,x):
        identity=x if self.downsample is None else self.downsample(x)

        x=self.relu(self.bn1(self.cn1(x)))
        x=self.bn2(self.cn2(x))+identity
        return self.relu(x)

class BottleneckBlock(nn.Module):

    expansion=4

    def __init__(self, ins, width, stride):
        super().__init__()

        self.stride=stride
        self.ins=ins
        self.width=width
        self.out=width*self.expansion
        self.relu=nn.ReLU(inplace=True)
    
        self.cn1=Conv1(ins=ins,out=width,stride=1)
        self.bn1=nn.BatchNorm2d(width)
        self.cn2=Conv3(ins=width,out=width,stride=stride)
        self.bn2=nn.BatchNorm2d(width)
        self.cn3=Conv1(ins=width,out=self.expansion*width,stride=1)
        self.bn3=nn.BatchNorm2d(self.expansion*width)

        if stride!=1 or self.ins!=self.out:
            self.downsample=nn.Sequential(Conv1(ins, self.out,stride), nn.BatchNorm2d(self.expansion*width))
        else:
            self.downsample=None

    def forward(self, x):
        identity=x if self.downsample is None else self.downsample(x)
        out=self.relu(self.bn1(self.cn1(x)))
        out=self.relu(self.bn2(self.cn2(out)))
        return(self.relu(self.bn3(self.cn3(out))+identity))

class ResNet(nn.Module):
    def __init__(self, blocktype, blocks_per_stage:tuple, num_classes:int, zero_init_residual:bool):
        super().__init__()

        if blocktype not in [BasicBlock, BottleneckBlock]:
            raise TypeError("Block type must be either 'BasicBlock' or 'BottleneckBlock'")

        self.outs_per_stage=[64,128,256,512]

        self.stem=nn.Sequential(
            Conv7(ins=3, out=64,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
            )

        layers=[]
        ins=64
        expansion=blocktype.expansion
        for stage in range(4):
            width=self.outs_per_stage[stage]
            for i in range(blocks_per_stage[stage]):
                stride=2 if (i==0 and stage!=0) else 1
                layers.append(blocktype(ins,width,stride))
                ins=expansion*width

        head=[]
        head.append(nn.AdaptiveAvgPool2d(1))
        head.append(nn.Flatten())
        head.append(nn.Linear(in_features=expansion*512, out_features=num_classes))

        self.block_stack=nn.Sequential(*layers)
        self.head_stack=nn.Sequential(*head)

        self._initialize_layers()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, BottleneckBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    def _initialize_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        out=self.stem(x)
        out=self.block_stack(out)
        out=self.head_stack(out)
        return(out)

def ResNet18(num_classes=1000, zero_init_residual=False):
    return ResNet(BasicBlock,(2,2,2,2),num_classes,zero_init_residual=zero_init_residual)

def ResNet34(num_classes=1000, zero_init_residual=False):
    return ResNet(BasicBlock,(3,4,6,3),num_classes,zero_init_residual=zero_init_residual)

def ResNet50(num_classes=1000, zero_init_residual=False):
    return ResNet(BottleneckBlock,(3,4,6,3),num_classes,zero_init_residual=zero_init_residual)

def ResNet101(num_classes=1000, zero_init_residual=False):
    return ResNet(BottleneckBlock,(3,4,23,3),num_classes,zero_init_residual=zero_init_residual)