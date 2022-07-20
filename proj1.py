from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from collections import OrderedDict
import time

f = open("lenet_cfg_4.txt", "w")
def printToFile(s):
    f.write(s)
    f.write("\n")

print = printToFile

# Preparing for Data
print('==> Preparing data..')

# Data augmentation
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

##
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        """ 
        1) FC with dropout, CONV with BN 
        2) FC with dropout, CONV without BN 
        3) FC without dropout, CONV with BN 
        4) FC without dropout, CONV without BN
        """
    
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1,6,kernel_size=(5,5), padding=2)),
            #('batch1', nn.BatchNorm2d(6)),                    ## BATCH NORMALIZATION LAYER
            ('tanh1', nn.Tanh()),
            ('s2', nn.AvgPool2d(kernel_size=(2,2), stride=2)),
            ('c3',nn.Conv2d(6, 16, kernel_size=(5,5))),
            #('batch2', nn.BatchNorm2d(16)),                    ## BATCH NORMALIZATION LAYER
            ('tanh3', nn.Tanh()),
            ('s4', nn.AvgPool2d(kernel_size=(2,2), stride=2)),
            ('c5',nn.Conv2d(16, 120, kernel_size=(5,5))),
        ]))
        
        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120,84)),
            ('tanh6', nn.Tanh()),
            #('drop1', nn.Dropout(p=0.5)),                           ## DROPOUT LAYER
            ('f7', nn.Linear(84,10)),
            ('sig7',nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, x):
    
        y = self.convnet(x)
        y = torch.flatten(y, 1)
        out = self.fc(y)

        return out



def train(model, device, train_loader, optimizer, epoch):
    model.train()
    count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test( model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    time0 = time.time()
    # Training settings
    batch_size = 128
    epochs = 10
    lr = 0.05
    no_cuda = True
    save_model = False
    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(100)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    trainset = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle = True)
    testset = datasets.MNIST('../data', train=False,
                       transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle = True)

    model = LeNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for epoch in range(1, epochs + 1):
        train( model, device, train_loader, optimizer, epoch)
        test( model, device, test_loader)

    if (save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
    time1 = time.time() 
    print ('Traning and Testing total excution time is: %s seconds ' % (time1-time0))   
if __name__ == '__main__':
    main()
    f.close()
