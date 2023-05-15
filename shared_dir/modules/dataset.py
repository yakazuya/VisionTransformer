import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import os
from PIL import Image
from pathlib import Path

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