import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os
import torch
import numpy as np
import pickle
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import copy
import torchvision
from PIL import Image
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import sys
from scipy import stats

########### this file takes the response of the trained models of all of the 4899 stimuli
########### and the ground truth firing response of the 227 neurons to the 4899 stimuli
########### Note, I did this with two models.  One with VGG16 chopped off after 17 layers
########### And another one with the VGG16 chopped off after 23 layers.  


predictions17 = pickle.load(open("layer17PredictionsAll.p", "rb"))
targets17 = pickle.load(open("layer17TargetsAll.p", "rb"))
predictions23 = pickle.load(open("layer23PredictionsAll.p", "rb"))
targets23 = pickle.load(open("layer23TargetsAll.p", "rb"))

orders = pickle.load(open("orderStimsLast.p", "rb"))
print(len(orders))
print(orders[0])
assert False

print(len(predictions17))   #4899
print(len(targets17))       #4899

predictions17 = np.asarray(predictions17)
targets17 = np.asarray(targets17)

############ this part of the code takes gets the 4 best stimuli (by index) for each neuron

differences17 = predictions17 - targets17
print(differences17.shape)  #4899 x 1 x 227
a = differences17.argsort(axis = 0)[:4]
print(a.shape)
print(a[0])

print("did this work?")
assert False

#############  Now I am going to go back and get the best stimuli (by name)



print(differences17[0])
differences17 = np.absolute(differences17)
print(differences17[0])

#differences17.min(axis=1)
#x = np.argmin(differences17, axis=0)

predictions23 = np.asarray(predictions23)
targets23 = np.asarray(targets23)

#print(predictions17[0])
#print(predictions17[0].shape)  #(1, 227)
#print(targets17[0])
#print(targets17[0][0][0])    #first stimuli, second is nothing, first neuron
#print(targets17[0].shape)     #(1, 227)
print(predictions17.shape)   #(1, 227)
print(predictions17[:, :, 1].shape)
assert False


correlations23 = []
correlations17 = []
for i in range(227):
    print(i)
    prediction1 = predictions17[:,:,i]
    target1 = targets17[:,:,i]
    target2 = targets23[:,:,i]
    prediction2 = predictions23[:,:,i] 
    (corr, _) = stats.pearsonr(prediction1, target1)
    correlations17.append(corr)
    (corr,_) = stats.pearsonr(prediction2, target2)
    correlations23.append(corr)

pickle.dump(correlations23, open("neuronCorr23.p", "wb"))
pickle.dump(correlations17, open("neuronCorr17.p", "wb"))
assert False



predictions17 = predictions17.flatten()   #0.625
targets17 = targets17.flatten()           #0.625 on a later iteration

predictions23 = predictions23.flatten()   #.674
targets23 = targets23.flatten()           #.674 on a later iteration


print(predictions23.shape)
print(targets23.shape)



#scipy.stats.pearsonr

(corr, _) = stats.pearsonr(predictions17, targets17)
print(corr)
assert False


#find . -maxdepth 2 -type f -name '.DS_Store*' -delete

labels = pickle.load(open("dataForCNNNorm1.p", "rb"))
images = pickle.load(open("imageDataRGB.p", "rb"), encoding = "bytes")

### make this work to shuffle so we can not overfit
#labels = np.repeat(labels, 24)   #29 is magic numbered in.  It is the number of electrodes
#print(labels.shape)

print(len(labels))
print(len(images))
print(labels[0].shape)
print(images[1].shape)
temp = []

for i in range(len(labels)):
    temp.append((labels[i], images[i]))


print(len(temp))  # good so far


temp = np.asarray(temp)


#labels = labels.reshape(labels.shape[0], 1)
#dataTemp = np.concatenate((images, labels), axis = 1)
#print(dataTemp.shape)
#assert False
np.random.shuffle(temp)
print(temp.shape)


labels = []
images = []

for i in range(len(temp)):
    labels.append(temp[i][0])
    images.append(temp[i][1])
    
   

print(len(labels))
print(len(images))



trainLabels = labels[:3500]
testLabels = labels[3500:]

trainImages = images[:3500]
testImages = images[3500:]

pickle.dump(trainLabels, open("dataForCNNNormTrain.p", "wb"))
pickle.dump(testLabels, open("dataForCNNNormTest.p", "wb"))
pickle.dump(trainImages, open("imageDataRGBTrain.p", "wb"))
pickle.dump(testImages, open("imageDataRGBTest.p", "wb"))

print("finished muffins")
assert False





print(len(labels))
print(type(labels))



row_sums = labels.sum(axis=1)
new_matrix = labels / row_sums[:, np.newaxis]
print(new_matrix.shape)

print(new_matrix[0])
pickle.dump(new_matrix, open("dataForCNNNorm1.p", "wb"))
assert False


train_data = torchvision.datasets.ImageFolder('./boldImages/train/', transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, batch_size= 10 , shuffle = True, num_workers =4)

val_data = torchvision.datasets.ImageFolder('./boldImages/validation/', transform = transforms.ToTensor())
val_loader = torch.utils.data.DataLoader(val_data, batch_size= 10 , shuffle = True, num_workers =4)

GPU_ID = int(sys.argv[1])
     
if GPU_ID !=-1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=2, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x, embedding = False):
        #print("hello")
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #print("hi hi")
        #assert False
        #print(x.shape)    #torch.Size([5, 25088])
       
        #assert False
        if embedding == True:
            return x

        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    #print("making layers")
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


net = vgg16_bn(pretrained = False) #.to(device)
#net = nn.Sequential(*list(net.children())[:-1], nn.Linear(7, 2))

if GPU_ID ==-1:
   net = nn.DataParallel(net)
net.cuda()


# Loss and optimizer
#criterion = AngleLoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=.001)
scheduler = ReduceLROnPlateau(optimizer)

# Train the model
#total_step = len(data_loader)

scheduler = ReduceLROnPlateau(optimizer)

# Train the model
#total_step = len(data_loader)

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    #model.to(device)

    running_loss = 0.0
    running_corrects = 0.0
    total = 0.0
    correct_predictions = 0.0

    #start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        #data = Variable(torcg,grinumpy (train_x).long to (device)long) 
        optimizer.zero_grad()
        #print("got here")
        data = data.to(device)
        target = target.to(device)  #this is the label

        outputs = model(data)  #this is the same as model.forward
        #net = nn.Sequential(*list(model.children())[:-1], nn.Linear(7, 2))
        #net.cuda()
        #outputs = net(outputs)
        #print(outputs.shape)   # (10,2)
        #assert False
        #print(outputs.shape)     #torch.Size([10, 512, 7, 2])
        #assert False
        #print(outputs.shape)   #[100, 512]
        #print(target.shape)    #[100]
        #assert False
        _, predicted = outputs.max(1)
        total += target.size(0)


        #total_predictions += target.size(0)
        correct_predictions += predicted.eq(target).sum().item()
        loss = criterion(outputs, target)
        #print(loss)
        #assert False
        running_loss += loss.item()
        #running_corrects += torch.sum(preds == target.data)
        #print(running_corrects)
        #assert False
        loss.backward()
        optimizer.step()





    acc = (correct_predictions/total) *100.0
    print('Training Loss: ', running_loss)#, 'Time: ',end_time - start_time, 's')
    print('Training Accuracy: ', acc)
    return running_loss




def val_epoch(model, val_loader):
    model.eval()
    #model.to(device)

    running_loss = 0.0
    running_corrects = 0.0
    total = 0.0
    correct_predictions = 0.0

    #start_time = time.time()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            #optimizer.zero_grad()   
            data = data.to(device)
            target = target.to(device)

            outputs = model(data)  #this is the same as model.forward
            _, predicted = outputs.max(1)
            #total += target.size(0) 


        #total_predictions += target.size(0)
            correct_predictions += predicted.eq(target).sum().item()
            #max_vals, preds = torch.max(outputs.data, 1)
            total += target.size(0)
            #correct_predictions += (preds ==target).sum().item()
            #print(outputs.shape)
            #, preds = torch.max(outputs, 1)
            loss = criterion(outputs, target)
            running_loss += loss.item()
            #running_corrects += torch.sum(preds == target.data)
    running_loss /= len(val_loader)
    acc = (correct_predictions/total) *100.0

    print('Validation Loss: ', running_loss) #, 'Time: ',end_time - start_time, 's')
    print('Validation Accuracy: ', acc)
    return running_loss




for i in range(30):
    print(i)
    print("train Example\n")
    trainloss= train_epoch(net,train_loader, criterion, optimizer)
    valloss = val_epoch(net, val_loader)
    path = "vgg16Epoch_" + str(i)
    torch.save(net.state_dict(), path)


print("all done")
assert False






'''
train_epoch(model, train_loader, criterion, optimizer):
    net.train()
    #model.to(device)

    running_loss = 0.0
    running_corrects = 0.0
    total = 0.0
    correct_predictions = 0.0
    
    #start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):  
        #data = Variable(torcg,grinumpy (train_x).long to (device)long) 
        optimizer.zero_grad()   
        data = data.to(device)
        target = target.to(device)  #this is the label

        outputs = net(data)  #this is the same as model.forward
        #print(outputs.shape)   #[100, 512]
        #print(target.shape)    #[100]
        #assert False
        _, predicted = outputs.max(1)
        total += target.size(0) 


        #total_predictions += target.size(0)
        correct_predictions += predicted.eq(target).sum().item()

        loss = criterion(outputs, target)
        running_loss += loss.item()
        #running_corrects += torch.sum(preds == target.data)
        #print(running_corrects)
        #assert False
        loss.backward()
        optimizer.step()
        #print('Epoch Accuracy:')
    
    #end_time = time.time()
    
    running_loss /= len(train_loader)
    acc = (correct_predictions/total) *100.0
    print('Training Loss: ', running_loss)#, 'Time: ',end_time - start_time, 's')
    print('Training Accuracy: ', acc)
    return running_loss
'''


for i in range(40):
    print(i)
    print("train Example\n")
    trainloss= train_epoch(net,train_loader, criterion, optimizer)
    path = "firstVGG16Epoch_" + str(i)
    torch.save(model.state_dict(), path)


