import torch
from torch import nn

class ResidualDownSamplingBlock(nn.Module):
    def __init__(self,in_chann,first=False):
        super(ResidualDownSamplingBlock,self).__init__()
        self.first=first
        if(first):
         self.conv1=nn.Conv2d(in_chann,in_chann,1,1)
         self.conv2=nn.Conv2d(in_chann,in_chann,3,1,1)
         self.conv3=nn.Conv2d(in_chann,in_chann*4,1,1)
         self.convShortcut=nn.Conv2d(in_chann,in_chann*4,1,1)

        else:
         self.conv1=nn.Conv2d(in_chann,in_chann//2,1,2)
         self.conv2=nn.Conv2d(in_chann//2,in_chann//2,3,1,1)
         self.conv3=nn.Conv2d(in_chann//2,in_chann*2,1,1)
         self.avgpooling=nn.AvgPool2d(2,2)
         self.convShortcut=nn.Conv2d(in_chann,in_chann*2,1,1)
        
        self.relu=nn.ReLU()
        self.batchN1=nn.BatchNorm2d(in_chann)
        self.batchN2=nn.BatchNorm2d(in_chann)
        self.batchN3=nn.BatchNorm2d(in_chann*4)
        self.batchN1_2=nn.BatchNorm2d(in_chann//2)
        self.batchN2_2=nn.BatchNorm2d(in_chann//2)
        self.batchN3_2=nn.BatchNorm2d(in_chann*2)
       
        
         
    def forward(self,x):
        
        
        if(self.first):
         out1=self.relu(self.batchN1(self.conv1(x)))
         out2=self.relu(self.batchN2(self.conv2(out1)))
         out3=self.batchN3(self.conv3(out2))
         shortcut=self.convShortcut(x)
        else:
         out1=self.relu(self.batchN1_2(self.conv1(x)))
         out2=self.relu(self.batchN2_2(self.conv2(out1)))
         out3=self.batchN3_2(self.conv3(out2))
         shortcut=self.convShortcut(x)
         shortcut=self.convShortcut(self.avgpooling(x))
        return self.relu(out3+shortcut)
    
    
    
    
class ResidualBasicBlock(nn.Module):
    def __init__(self,in_chann):
        super(ResidualBasicBlock,self).__init__()
        self.conv1=nn.Conv2d(in_chann,in_chann,3,1,1)
        self.relu=nn.ReLU()
        self.batchN1=nn.BatchNorm2d(in_chann)
        self.batchN2=nn.BatchNorm2d(in_chann)
        
        self.conv2=nn.Conv2d(in_chann,in_chann,3,1,1)
        
        
    def forward(self,x):
        out1=self.relu(self.batchN1(self.conv1(x)))
        out2=self.batchN2(self.conv2(out1))
        
        shortcut=x
        return self.relu(out2+shortcut)
        
        

        

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet,self).__init__()
        self.stem = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
)
        self.stack=nn.Sequential(
            self.stem,
            ResidualDownSamplingBlock(64,True),
            ResidualBasicBlock(256),
            ResidualBasicBlock(256),
            ResidualDownSamplingBlock(256),
            ResidualBasicBlock(512),
            ResidualBasicBlock(512),
            ResidualDownSamplingBlock(512),
            ResidualBasicBlock(1024),
            ResidualBasicBlock(1024),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1024,200)
            
        
            
        )
    def forward(self,x):
        x=self.stack(x)
        return x
    
    
if __name__ == "__main__":
    model = CustomNet()
    x = torch.randn(4, 3, 64, 64)  # 4 immagini finte 64x64
    out = model(x)
    print("Output shape:", out.shape)  # Deve stampare: torch.Size([4, 200])