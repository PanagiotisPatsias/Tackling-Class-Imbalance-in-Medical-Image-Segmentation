import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(inc, outc, 3, padding=1, bias=False),
            nn.InstanceNorm2d(outc),
            nn.ReLU(inplace=True),
            nn.Conv2d(outc, outc, 3, padding=1, bias=False),
            nn.InstanceNorm2d(outc),
            nn.ReLU(inplace=True))
    def forward(self,x):
        return self.op(x)


class Down(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.mp = nn.MaxPool2d(2)
        self.conv = DoubleConv(inc,outc)
    def forward(self,x):
        return self.conv(self.mp(x))


class Up(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.up = nn.ConvTranspose2d(inc, inc//2, 2, 2)
        self.conv = DoubleConv(inc, outc)
    def forward(self,x1,x2):
        x1 = self.up(x1)
        diffY = x2.size()[2]-x1.size()[2]
        diffX = x2.size()[3]-x1.size()[3]
        x1 = nn.functional.pad(x1,[diffX//2,diffX-diffX//2,diffY//2,diffY-diffY//2])
        return self.conv(torch.cat([x2,x1],1))


class UNet2D(nn.Module):
    def __init__(self, in_channels=3, out_classes=1, base=64):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.d1 = Down(base, base*2)
        self.d2 = Down(base*2, base*4)
        self.d3 = Down(base*4, base*8)
        self.d4 = Down(base*8, base*16)
        self.u1 = Up(base*16, base*8)
        self.u2 = Up(base*8, base*4)
        self.u3 = Up(base*4, base*2)
        self.u4 = Up(base*2, base)
        self.outc = nn.Conv2d(base, out_classes, 1)
        self._init()
    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
    def forward(self,x):
        x1=self.inc(x)
        x2=self.d1(x1)
        x3=self.d2(x2)
        x4=self.d3(x3)
        x5=self.d4(x4)
        x=self.u1(x5,x4)
        x=self.u2(x,x3)
        x=self.u3(x,x2)
        x=self.u4(x,x1)
        return self.outc(x) # logits