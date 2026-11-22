import os
import torch
import torch.nn as nn
import torch.optim as optim

from models.resnetImageNet200 import CustomNet
from data.custom4imagenetVal import get_dataloadersImagenet200
import train
from eval import validate

def main():
    data_root = "/content/tiny-imagenet-200"
    batch_size=128
    num_epochs=50
    lr=0.001
    save_path = "/content/drive/MyDrive/best_customnet.pth"
    if not torch.cuda.is_available():
        raise RuntimeError("GPU non attiva! Su Colab vai su Runtime -> Cambia tipo di runtime -> GPU.")
    device = torch.device("cuda")
    print(f"Training avviato su: {torch.cuda.get_device_name(0)}")
    train_loader, val_loader = get_dataloadersImagenet200(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=2
    )
    
    model=model = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoca {epoch}/{num_epochs} (LR: {optimizer.param_groups[0]['lr']:.6f}) ---")
        train(epoch, model, train_loader, criterion, optimizer)
        val_acc = validate(model, val_loader, criterion)
        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"--> Nuovo record! Modello salvato su Drive (Val Acc: {best_acc:.2f}%)")
    print(f"\nAddestramento completato! Miglior Validation Accuracy: {best_acc:.2f}%")
    
    
if __name__=="__main__":
    main()