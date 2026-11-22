import os
import pandas as pd
import torchvision
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def get_transforms_imagenet200():
    return {
    'train' : T.Compose([
        T.RandAugment(),
        T.RandomCrop(64,padding=8),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
    ]),
    'val' : T.Compose([
        
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
        
    ])
    
}

class DatasetImagenetVal200(Dataset):
    def __init__(self,annotations_file,img_dir,class_to_idx,transform=None,target_transform=None):
        self.img_labels=pd.read_csv(annotations_file, sep='\t',header=None)
        self.img_dir=img_dir
        self.transform=transform
        self.target_transform=target_transform
        self.class_to_idx=class_to_idx
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self,idx):
        img_path=os.path.join(self.img_dir,self.img_labels.iloc[idx,0])
        image=Image.open(img_path).convert('RGB')
        label=self.class_to_idx[self.img_labels.iloc[idx,1]]
        if(self.transform):
            image=self.transform(image)
        if self.target_transform:
            label=self.target_transform(label)
        return image,label
        
        
        


def get_dataloadersImagenet200(data_root, batch_size=128, num_workers=2):
    transforms = get_transforms_imagenet200()

    # 1. ImageFolder crea in automatico il dizionario train_dataset.class_to_idx
    train_dataset = ImageFolder(
        root=os.path.join(data_root,'train'),
        transform=transforms['train']
    )

    # 2. Passi train_dataset.class_to_idx alla tua classe custom per il validation
    val_dataset = DatasetImagenetVal200(
        annotations_file=os.path.join(data_root,'val','val_annotations.txt'),
        img_dir=os.path.join(data_root, 'val', 'images'),
        class_to_idx=train_dataset.class_to_idx,
        transform=transforms['val']
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,persistent_workers=True)

    return train_loader, val_loader
        
    