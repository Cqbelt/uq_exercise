import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import  Variable
import torchvision.models as models
import matplotlib.pyplot as plt
import time
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='use resnet18 to train cifar10')

parser.add_argument('--path', type=str,  default='./data',
                    help='datasets path')
parser.add_argument('--epochs', type=int, default=25,
                    help='How many epochs to train. Default: 60.')
parser.add_argument('--lr', type=float, nargs='?', action='store', default=1e-3,
                    help='learning rate. Default: 1e-3.')

args = parser.parse_args()
path = args.path
epochs = args.epochs

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

cifar_norm_mean = (0.49139968, 0.48215827, 0.44653124)
cifar_norm_std = (0.24703233, 0.24348505, 0.26158768)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar_norm_mean, cifar_norm_std),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_norm_mean, cifar_norm_std),
])

trainset = torchvision.datasets.CIFAR10(root=path, train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=path, train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
           

class_num = 10
resnet18 = models.resnet18(pretrained=False)
resnet18.load_state_dict(torch.load('resnet18.pth'))
channel_in = resnet18.fc.in_features
resnet18.fc = nn.Linear(channel_in,class_num)

for param in resnet18.parameters():
    param.requires_grad = False

for param in resnet18.fc.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
                        filter(lambda p: p.requires_grad, resnet18.parameters()),#重要的是这一句
                        lr=0.1,momentum=0.9)
                        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18 = resnet18.to(device)

#inputs = torch.rand(4,3,32,32)
#outputs = resnet18(inputs)
#print(outputs.size())

def test():
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = Variable(images)
            labels = Variable(labels).to(device)
            outputs = resnet18(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
                        
            c = (predicted == labels).squeeze()
            for i in range(16):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    
def save_model(model):
    torch.save(model.state_dict(), '/output/final_net.pth')
    print('save model!')

def plot(lossdata):
    textsize = 15
    marker = 5
    plt.figure(dpi=100)
    x = list(range(len(lossdata)))
    plt.plot(x,lossdata , 'b-')
    fig = plt.gcf()
    print('losses_len:',len(lossdata))
    plt.savefig('/output/loss.jpg')
    plt.show()
        
def main():
    losses = []
    for epoch in range(epochs):  # 多批次循环
        toc0 = time.time()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 梯度置0
            optimizer.zero_grad()

            # 正向传播，反向传播，优化
            outputs = resnet18(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 打印状态信息
            running_loss += loss.item()
            if i % 1000 == 999:    # 每1000批次打印一次
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                losses += list(running_loss)
                running_loss = 0.0
        toc1 = time.time()
        print('time:',round(toc1-toc0,2))
    print('Finished Training')
    test()
    plot(losses)
    save_model(resnet18)
    
if __name__ == '__main__':
    main()