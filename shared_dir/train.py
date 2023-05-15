#! /usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from modules.dataset import CustomDataset
from modules.ViT import ViT
import os

def main():
    os.makedirs('/home/pytorch/weight', exist_ok=True)
    root_path = "/home/pytorch/PetImages"
    custom_train_dataset = CustomDataset(root_path, train=True)
    train_loader = torch.utils.data.DataLoader(dataset=custom_train_dataset,
                                               batch_size = 5,
                                               shuffle=True)
    custom_val_dataset = CustomDataset(root_path, train=False)
    val_loader = torch.utils.data.DataLoader(dataset=custom_val_dataset,
                                            batch_size = 5,
                                            shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ViT(
        in_channels=3,
        num_classes=2,
        emb_dim=1024,
        num_patch_row=16,
        image_size=224,
        num_blocks=24,
        head=16,
        hidden_dim=1024*4,
        dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    num_epoch = 1000
    val_acc_max = 0

    for epoch in range(num_epoch):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        # train
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            train_acc += (outputs.max(1)[1] == labels).sum().item()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_acc / len(train_loader.dataset)

        # val
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.max(1)[1] == labels).sum().item()
        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_acc = val_acc / len(val_loader.dataset)

        if val_acc_max < avg_val_acc:
            val_acc_max=avg_val_acc
            torch.save(model.state_dict(), '/home/pytorch/weight/ViT.pth')

        print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}')
    print(f'val_acc_max: {val_acc_max}')

if __name__ == '__main__':
    main()