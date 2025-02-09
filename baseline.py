import os
from sched import scheduler
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import SGD

# 定义数据集路径
train_dir = './dataset1/train'
val_dir = './dataset1/val'
test_dir = './dataset1/test'


# 自定义数据集
class ImageDataset(Dataset):
    def __init__(self, uav_dir, background_dir, transform=None):
        self.uav_images = [os.path.join(uav_dir, img) for img in os.listdir(uav_dir)]
        self.background_images = [os.path.join(background_dir, img) for img in os.listdir(background_dir)]
        self.images = self.uav_images + self.background_images
        self.labels = [1] * len(self.uav_images) + [0] * len(self.background_images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

# 数据增强和转换
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
])

# 创建数据集和数据加载器
train_dataset = ImageDataset(os.path.join(train_dir, 'UAV'), os.path.join(train_dir, 'background'), transform=transform)
val_dataset = ImageDataset(os.path.join(val_dir, 'UAV'), os.path.join(val_dir, 'background'), transform=transform)
test_dataset = ImageDataset(os.path.join(test_dir, 'UAV'), os.path.join(test_dir, 'background'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

#::

device='cuda'
class mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18()
        self.fc = nn.Linear(1000,1)
        self.AF=nn.Sigmoid()
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        x = self.AF(x)
        return x

model=mymodel().to(device)

loss_fn = nn.BCELoss()
optimizer = SGD(model.parameters(), lr=1e-2)
scheduler1=optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=0,threshold=0.001,min_lr=1e-5,threshold_mode='abs')

state_dict = torch.load('resnet18_weight.pth')
print("loading success")
model.load_state_dict(state_dict)
model.to(device)

def train_batch(x, y, model, optimizer, loss_fn):

    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()

@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    is_correct = abs(prediction-y)<0.5
    return is_correct.cpu().numpy().tolist()
import time
losses, accuracies = [], []
for epoch in range(10):
    print(epoch)
    start=time.time()
    epoch_losses, epoch_accuracies = [], []
    for idx, batch in enumerate(iter(train_loader)):
       ima, lab, path = batch
       lab = lab.unsqueeze(1)
       batch_loss = train_batch(ima.float().to(device),lab.float().to(device), model, optimizer, loss_fn)
       epoch_losses.append(batch_loss)
    end=time.time()
    print("time=",end-start,"s")
    epoch_loss = np.array(epoch_losses).mean()
    print("epoch_loss:",epoch_loss)
    for idx, batch in enumerate(iter(train_loader)):
        ima, lab, path = batch
        lab = lab.unsqueeze(1)
        is_correct = accuracy(ima.float().to(device), lab.float().to(device), model)
        epoch_accuracies.extend(is_correct)
    epoch_accuracy = np.mean(epoch_accuracies)
    print("epoch_accuracy:",epoch_accuracy)

    losses.append(epoch_loss)
    accuracies.append(epoch_accuracy)

    save_path = 'resnet18_weight.pth'
    torch.save(model.state_dict(), save_path)
    print("save success,save path:",save_path)

epochs = np.arange(10)+1
plt.figure(figsize=(20,5))
plt.subplot(121)
plt.title('Loss value over increasing epochs')
plt.plot(epochs, losses, label='Training Loss')
plt.legend()
plt.subplot(122)
plt.title('Accuracy value over increasing epochs')
plt.plot(epochs, accuracies, label='Training Accuracy')
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.legend()
plt.show()