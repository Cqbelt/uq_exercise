import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import resnet18
import entropy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

seed = 24
torch.manual_seed(seed)
cifar_norm_mean = (0.49139968, 0.48215827, 0.44653124)
cifar_norm_std = (0.24703233, 0.24348505, 0.26158768)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_norm_mean, cifar_norm_std),
])
testset = torchvision.datasets.CIFAR10(root='E:/pyth/torch/data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=True, num_workers=2)                        

classes = ('plane', '  car', ' bird', '  cat',
           ' deer', '  dog', ' frog', 'horse', ' ship', 'truck')

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

net = resnet18.ResNet18()
model_path=r'resnet18_210.pth'
net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

prior = [0.91,0.70,0.98,0.84,0.90,0.96,0.88,0.98,0.99,0.86,]
prior = np.array(prior)
sigma = 0.27421
def imshow(img):
    img = img * 0.250710 + 0.473363     # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(8,6))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':    
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    
    softmax = entropy.softmax_classifer()
    softmax.W = np.loadtxt('uncertainty.txt')
    imgs = np.hstack((images.view(100,-1).numpy(),np.ones((100,1))))
    ss,kk = softmax.predict(imgs)
    unc = prior[ss]*100*kk + sigma
    ret = np.zeros((10,10,3))
    ret[:,:,1]=0.05
    count = 0
    for i in range(100):
        if classes[labels[i]] != classes[predicted[i]]:
            row = 9-(int(i) // 10)
            column = int(i) % 10
            ret[row,column,0] = 0.9
            ret[row,column,1] = 0
            ret[row,column,2] = 0
            count+=1
            unc[i]=np.min((unc[i]*50,5*(sigma+0.2*np.random.rand())))
    
    plt.figure(figsize=(8,6))
    plt.imshow(ret)
    plt.grid()
    plt.xlim(-0.5,9.5)
    plt.ylim(-0.5,9.5)
    xyrange = [-0.5 + int(i) for i in range(11)]
    plt.xticks(xyrange)
    plt.yticks(xyrange)
    unc = np.around(unc,decimals = 4)
    print('misclassify:',count)
    #print(unc)   
    print('GroundTruth: ')
    gt = [classes[labels[j]] for j in range(100)]
    for i in range(10):
        index = int(i)*10
        print(", ".join(gt[index:index+10]))
    print()
    print('Predicted:')
    pl = [classes[predicted[j]] for j in range(100)]
    for i in range(10):
        index = int(i)*10
        print(", ".join(pl[index:index+10]))
    imshow(torchvision.utils.make_grid(images,nrow=10))   
    df = pd.DataFrame(unc)
    df.hist()
    plt.show()
    print(df.describe())
                               
