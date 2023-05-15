#モジュール読み込みを行う
#pytorch関連
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models ,transforms
from torch.utils.tensorboard import SummaryWriter

#画像読み込み関連
import numpy as np
from PIL import Image
import os
from pathlib import Path

#データ読み込み関連
#データセットクラス作成
class CustomDataset(torch.utils.data.Dataset):
    classes = ['cat', 'dog']

    def __init__(self, root, train=True):
        self.images = []
        self.labels = []

        root_path = root
        data_transforms = {
            'train' : transforms.Compose([
                transforms.RandomResizedCrop(size = (224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ]),
            'val' : transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ])
        }
        if train == True:
            root_cat = Path(root_path) / 'Cat/train'
            root_dog = Path(root_path) / 'Dog/train'
            self.transform = data_transforms['train']
            print(root_cat)
        else:
            root_cat = Path(root_path) / 'Cat/val'
            root_dog = Path(root_path) / 'Dog/val'
            self.transform = data_transforms['val']

        cat_list = list(Path(root_cat).glob('*.jpg'))
        dog_list = list(Path(root_dog).glob('*.jpg'))

        cat_labels = [0] * len(cat_list)
        dog_labels = [1] * len(dog_list)
        for image, label in zip(cat_list, cat_labels):
            self.images.append(image)
            self.labels.append(label)
        for image, label in zip(dog_list, dog_labels):
            self.images.append(image)
            self.labels.append(label)
        
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        with open(str(image), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        

        return img, label

    def __len__(self):
        return len(self.images)

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

class AlexNet(nn.Module):
   def __init__(self,num_classes,fc_size):
      super(AlexNet,self).__init__()
      #畳み込み関連
      self.features = nn.Sequential(
         nn.Conv2d(3,64,kernel_size=11,stride=4,padding=2),
         nn.ReLU(inplace=True),
         nn.MaxPool2d(kernel_size=2,stride=2),
         nn.Conv2d(64,192,kernel_size=5,padding=2),
         nn.ReLU(inplace=True),
         nn.MaxPool2d(kernel_size=2,stride=2),
         nn.Conv2d(192,384,kernel_size=3,padding=1),
         nn.ReLU(inplace=True),
         nn.Conv2d(384,256,kernel_size=3,padding=1),
         nn.ReLU(inplace=True),
         nn.Conv2d(256,256,kernel_size=3,padding=1),
         nn.ReLU(inplace=True),
         nn.MaxPool2d(kernel_size=2,stride=2),
      )
      #全結合関連
      self.classifier =nn.Sequential(
         nn.Dropout(p=0.5),
         nn.Linear(fc_size,4096),
         nn.ReLU(inplace=True),
         nn.Dropout(p=0.5),
         nn.Linear(4096,4096),
         nn.ReLU(inplace=True),
         nn.Linear(4096,num_classes)
      )

   def forward(self,x):
      #畳み込み層の結果
      x= self.features(x)
      #縦xよこx奥行きからベクトルに変更
      x=x.view(x.size(0),-1)
      #全結合の結果
      x= self.classifier(x)
      return x
   
num_classes =2
fc_size=6*6*256

device='cuda' if torch.cuda.is_available() else 'cpu'
net = AlexNet(num_classes,fc_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer =optim.SGD(net.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4)

num_epochs=100
# writer=SummaryWriter(log_dir='./logs')
val_acc_max=0
for epoch in range(num_epochs):
       train_loss=0
       train_acc=0
       val_loss=0
       val_acc=0

       #train
       net.train()
       for i, (images,labels) in enumerate(train_loader):
         images,labels=images.to(device), labels.to(device)

         optimizer.zero_grad()
         outputs=net(images)
         loss =criterion(outputs,labels)
         train_loss += loss.item()
         train_acc += (outputs.max(1)[1]==labels).sum().item()
         loss.backward()
         optimizer.step()
      
       avg_train_loss =train_loss / len(train_loader.dataset)
       avg_train_acc = train_acc / len(train_loader.dataset)

       #val
       net.eval()
       
       with torch.no_grad():
         for images,labels in val_loader:
          images=images.to(device)
          labels=labels.to(device)
          outputs=net(images)
          loss=criterion(outputs,labels)
          val_loss +=loss.item()
          val_acc +=(outputs.max(1)[1]==labels).sum().item()

       avg_val_loss =val_loss/len(val_loader.dataset)
       avg_val_acc=val_acc/len(val_loader.dataset)
       
       if val_acc_max < avg_val_acc:
          val_acc_max=avg_val_acc
          torch.save(net.state_dict(),'/home/pytorch/weight/Alexnet.pth')
       print('Epoch [{}/{}],LOSS:{loss:.4f},val_loss:{val_loss:.4f},val_acc:{val_acc:.4f}'
               .format(epoch+1,num_epochs,i+1,loss=avg_train_loss,val_loss=avg_val_loss,val_acc=avg_val_acc))
    #    writer.add_scalar('train_loss',avg_train_loss,epoch)
    #    writer.add_scalar('train_acc',avg_train_acc,epoch)
    #    writer.add_scalar('val_loss',avg_val_loss,epoch)
    #    writer.add_scalar('val_acc',avg_val_acc,epoch)
       
    #    img_grid=torchvision.utils.make_grid(images)
    #    writer.add_image('input_image',img_grid,epoch)
print(f'val_acc_max: {val_acc_max}')
   
   
   


