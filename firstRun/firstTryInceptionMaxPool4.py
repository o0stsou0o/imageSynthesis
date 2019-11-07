"""
Various functions that use the Optimizer class to
do some common tasks.
"""
import os
import torch
import numpy as np

from losses import *
from utils import *
from functools import partial
from optimizer import Optimizer
#from scikit-image import imread, imsave
from skimage.io import imread, imsave
from torchvision.models import (alexnet, resnet50, vgg16)
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


net = vgg16(pretrained = True) #.to(device)


moduleList = list(net.features.modules())

net.features = nn.Sequential(*moduleList[0][:24])
features = list(net.children())[:-7] # Remove last layer
myLayer = nn.Linear(25088, 227)

features.extend([myLayer]) # Add our layer with 4 outputs
net.classifier = nn.Sequential(*features)

print(net)


def visualize(network, layer, idx, img_shape=(3, 224, 224),
        init_range=(0, 1), max_iter=400, lr=1, sigma=0,
        debug=False):
    """
    Perform standard Deep Dream-style visualization on the
    network.
    
    Parameters:

    network: the network to be run, a torch module

    layer: the layer of the network that the desired neuron
    to visualize is part of, also a torch module

    idx: a tuple of indexes into the output of the given
    layer (like (0,0,0,0) for a BCHW conv layer) that
    extracts the desired neuron

    img_shape: a tuple specifying the shape of the images the
    network takes in, in CHW form (a batch dimension is
    expected by Optimizer and so is automatically added)

    init_range: the range of values to randomly initialize
    the image to

    max_iter: the maximum number of iterations to run
    the optimization for

    lr: the 'learning rate' (the multiplier of the gradient
    as it's added to the image at each step; 

    sigma: the standard deviation (or list of stddevs)of 
    a Gaussian filter that smooths the image each iteration,
    standard for inception loop-style visualization

    debug: prints loss at every iteration if true, useful for
    finding the right learning rateo
    
    Returns:

    optimized image
    loss for the last iteration
    """
    # partial application, since the index can't be passed in optimizer code
    loss_func = partial(specific_output_loss, idx=idx)
    optimizer = Optimizer(network, layer, loss_func)

    # set the 'target' optimization to be very high -- just want
    # to increase it as much as possible
    # since optimizer is actually gradient descent, make it negative
    # TODO: allow selection of populations, not just single neurons
    target_shape = (1,) 
    target = torch.ones(*target_shape) * 100
    target = target.cuda()

    # now start optimization
    rand_img = torch_rand_range(img_shape, init_range).unsqueeze(0).cuda()
    return optimizer.optimize(rand_img, target, max_iter=max_iter,
            lr=lr, sigma=sigma, debug=debug)




images = []

for i in range(227):
    print(i)
    (img, loss) = visualize(net.cuda(), myLayer.cuda(), (0,i), lr=10)
    images.append(img)



pickle.dump(images, open("firstImage23HighLR.p", "wb"))


print("wow this worked!")
assert False



others = []

(img2, loss) = visualize(net, myLayer, 4)

(img3, loss) = visualize(net, myLayer, 5)

(img4, loss) = visualize(net, myLayer, 30)

others.append(img2)
others.append(img3)
others.append(img4)


pickle.dump(img, open("firstImage.p", "wb"))
assert False








def gen_one_image(network, layer, image, noise_level, 
        loss_func, constant_area=0, max_iter=1000,
        lr=np.linspace(10, 0.5, 1000), sigma=0, grayscale=False,
        debug=False):
    """
    Generate a single modified stimulus from a source image.
    (This function is primarily for use by other wrappers).

    Parameters:

    layer: the actual layer object, part of the network, that
    you're extracting features from for the generation

    image: a single image, in BCHW format, on the same device
    as the network (for now just GPU)

    grayscale: whether or not the optimization should be done in
    grayscale (enforcing the RGB channels stay the same)

    other arguments are same as std_generate
    """
    # constant_area's default is actually dependent on image
    # so 0 there is just a non-None placeholder
    # set to the center (max_dim / 5) pixels by default
    if constant_area == 0:
        h_center = int(image.shape[2] / 2)
        w_center = int(image.shape[3] / 2)

        h_span = int(image.shape[2] / 10)
        w_span = int(image.shape[3] / 10)

        constant_area = (h_center - h_span, h_center + h_span,
                w_center - w_span, w_center + w_span)

    with torch.no_grad():
        acts = []
        hook = layer.register_forward_hook(
                lambda m,i,o: acts.append(o))

        _ = network(image)

        act = acts[0]
        hook.remove()

    noisy_act = add_noise(act, noise_level)

    optimizer = Optimizer(network, layer, loss_func)

    new_img, loss = optimizer.optimize(image, noisy_act,
            constant_area=constant_area, max_iter=max_iter,
            lr=lr, sigma=sigma, clip_image=True, 
            grayscale=grayscale, debug=debug)

    return new_img.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze(), loss


def std_generate(net_name, lay_idx, images, noise_level, 
        constant_area=0, max_iter=1000,
        lr=np.linspace(10, 0.5, 1000), sigma=0, debug=False,
        alpha=2, beta=2, lambda_a=0, lambda_b=0):
    """
    Standard stimulus generation, using torchvision models.

    Parameters:

    net_name: whether to use alexnet, resnet50, or vgg16

    lay_idx: the layer you want (counting convolutional and linear
    layers; e.g. resnet50 has about 50)

    images: a directory containing only images, the ones
    the network will run on

    noise_level: standard deviation for the gaussian noise
    to add to the intermediate representation

    constant_area: the area of the image to keep constant at
    each iteration ((h1, h2, w1, w2) indices), defaults to
    the center 20%

    max_iter: the maximum number of iterations to run,
    set to a reasonable default

    lr: the 'learning rate', the multiplier of the gradient
    when added to the image, can vary a lot depending on the
    scale of the image pixel values and the 

    sigma: the standard deviation of a gaussian used to smooth
    the generated image at each timestep, as regularization
    (0 means no smoothing)

    alpha: the value of alpha for the alpha-norm loss term

    beta: the value of beta for the total variation loss term

    lambda_a: the weight for the alpha-norm loss term

    lambda_b: the weight for the beta-norm loss term

    Returns:

    nothing, but in a new directory (net_name + images), a
    modified form of each image is saved
    """
    network = {
            'alexnet' : alexnet,
            'resnet50' : resnet50,
            'vgg16' : vgg16,
            }[net_name](pretrained=True).cuda()

    # treat each Conv2d or Linear module as a single layer
    layers = [l for l in get_atomic_layers(network)
            if isinstance(l, torch.nn.modules.conv.Conv2d)
            or isinstance(l, torch.nn.modules.linear.Linear)]
    layer = layers[lay_idx]


    # make a directory for the new images if it doesn't already exist
    try:
        os.mkdir(f"modified_{images}")
    except FileExistsError:
        # it already exists; just add to it
        pass

    loss_func = partial(standard_loss, alpha=alpha, beta=beta,
            lambda_a = lambda_a, lambda_b = lambda_b)

    if isinstance(images, str):
        # assume it's a directory of only images
        # TODO: be more flexible here
        files = os.listdir(images)

        for fname in files:
            img, grayscale = load_img_torchvision(images + '/' + fname)
            img = img.cuda()

            new_img, loss = gen_one_image(network, layer,
                    img, noise_level, loss_func,
                    constant_area, max_iter, lr, sigma, grayscale,
                    debug)

            imsave(fname=f"modified_{images}/{net_name}_{lay_idx}_{noise_level}_{fname}", arr=new_img)
    else:
        # TODO: maybe make this work with a list of already-loaded images
        raise NotImplementedError
