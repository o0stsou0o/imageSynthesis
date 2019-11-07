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

#################################################################################################################  
## This model is training a VGG with transfer learning with sparse l1 norm on last layer 
#######  imageDataRGB.p are the rgb images from bold5000 sorted
#######  dataForCNN.p are the average neural firing for the neurons for each stimuli 227 neurons 4899 stimuli####
### Alpha hyperparameter must be played around with for best performance.  Once this is trained, we do inception  ########
###################################################################


'''
GPU_ID = int(sys.argv[1])
     
if GPU_ID !=-1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

'''




class loader(Dataset):
    def __init__(self, text_mode = False):
        self.target = pickle.load(open('dataForCNNNorm1.p' , 'rb' ))
        self.data = pickle.load(open('imageDataRGB.p', 'rb'), encoding = 'bytes')
        self.transformData = transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
        #self.transformData = transforms.Compose([transforms.ToTensor()])  
        self.transformLabel = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        data = self.data[index]
        target = self.target[index]
        pf0 = self.transformData(data)
        target = torch.from_numpy(target)
        return pf0, target

    def __len__ (self):
        return len(self.target)

test_data = loader(text_mode = False)

test_loader = DataLoader(test_data, batch_size = 1, shuffle = False, num_workers = 4, drop_last=False)



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

    def __init__(self, features, num_classes=1000, init_weights=True):
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
    print("making layers")
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


net = vgg16(pretrained = True) #.to(device)
print(net)
print("this is the stoppage")
moduleList = list(net.features.modules())



net.features = nn.Sequential(*moduleList[0][:24])
features = list(net.children())[:-7] # Remove last layer
myLayer = nn.Linear(25088, 227)

features.extend([myLayer]) # Add our layer with 4 outputs
net.classifier = nn.Sequential(*features)


newModules = list(net.modules())

print(net)
net.eval()
net.load_state_dict(torch.load("epoch_99_.001_23LayerNew"))


net.cuda()


criterion = nn.MSELoss()

optimizer = torch.optim.Adam(myLayer.parameters(), lr =.00001)
scheduler = ReduceLROnPlateau(optimizer)



####### Definining A Hook now
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()




'''
if GPU_ID == -1:
    net = nn.DataParallel(net)
#net.cuda()
'''




def test_epoch(model, test_loader):
    model.eval()

    running_loss = 0.0
    running_corrects = 0.0
    total = 0.0
    correct_predictions = 0.0
    predictions = []
    targets = []
    #start_time = time.time()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):

            data = data.to(device)
            target = target.to(device)

        
            outputs = model(data)
            predictions.append(outputs)
            targets.append(target)


    return predictions, targets








predictions, targets = test_epoch(net, test_loader)

print(predictions[0])
print(type(predictions[0]))
print(targets[0])
print(type(targets[0]))


finalPredictions = []
finalTargets = []

for item in predictions:
    finalPredictions.append(item.cpu().detach().numpy())

for item in targets:
    finalTargets.append(item.cpu().detach().numpy())



pickle.dump(finalPredictions, open("layer23PredictionsAll.p" , "wb"))
pickle.dump(finalTargets, open("layer23TargetsAll.p", "wb"))


print("done with all of it")
