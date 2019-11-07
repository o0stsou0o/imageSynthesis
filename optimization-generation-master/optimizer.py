import torch
import numpy as np
from utils import *

class Optimizer():
    """
    Optimize an image to produce some result in a deep net.
    """

    def __init__(self, net, layer, loss_func):
        """
        Parameters:

        net: nn.Module, presumably a deep net
        layer: nn.Module, part of the network that gives relevant output
        loss_func: callable taking layer output, target output, and image,
        returning the loss
        """
        super().__init__()

        self.net = net
        self.layer = layer
        self.loss_func = loss_func

        # will only define hooks during optimization so they can be removed
        self.acts = []
        self.grads = []

    def optimize(self, image, target, constant_area=None, max_iter=1000,
            lr=np.linspace(5, 0.5, 1000), clip_image=False,
            grayscale=False, sigma=0, debug=False):
        """
        Parameters:

        image: image to start from, presumably where the target was 
        modified from

        target: target activation, to be passed into loss_func

        constant_area: indices such that image[0:1, 2:3, :] stays
        constant each iteration of gradient ascent
        
        max_iter: maximum number of iterations to run

        lr: 'learning rate' (multiplier of gradient added to image at
        each step, or iterable of same length as max_iter with varying values)

        clip_image: whether or not to clip the image to real (0-256) pixel
        values, with standard torchvision transformations

        sigma: sigma of the Gaussian smoothing at each iteration
        (default value 0 means no smoothing), can be an iterable of
        length max_iter like 'lr'

        debug: whether or not to print loss each iteration

        Returns:

        optimized image
        loss for the last iteration
        """
        image.requires_grad_(False)
        new_img = torch.tensor(image, requires_grad=True)

        # change it to an array even if it's constant, for the iterating code
        if isinstance(lr, int):
            lr = [lr] * max_iter

        if isinstance(sigma, float) or isinstance(sigma, int):
            sigma = [sigma] * max_iter

        # want the actual, atomic first layer object for hooking
        children = [child for child in self.net.modules()
                if len(list(child.children())) == 0]

        # set up hooks
        back_hook = children[0].register_backward_hook(
                lambda m,i,o: self.grads.append(i[0]))

        forw_hook = self.layer.register_forward_hook(
                lambda m,i,o: self.acts.append(o))

        # now do gradient ascent
        for i in range(max_iter):
            # get gradient
            _ = self.net(new_img)
            loss = self.loss_func(self.acts[0], target, new_img)
            self.net.zero_grad()
            loss.backward()

            if debug:
                print(f'loss for iter {i}: {loss}')

            # all processing of gradient was done in loss_func
            # even momentum if applicable; none is done here
            with torch.no_grad():
                new_img.data = new_img.data - lr[i] * self.grads[0].data

                if clip_image:
                    new_img.data = clip_img(new_img.data)

                if sigma[i] > 0:
                    new_img.data = torch_gaussian_filter(new_img, sigma[i])

                if constant_area is not None:
                    # assuming torchvision structure (BCHW) here
                    # TODO: change this
                    new_img.data[:, :, constant_area[0]:constant_area[1],
                            constant_area[2]:constant_area[3]] = image[:, :,
                                    constant_area[0]:constant_area[1],
                                    constant_area[2]:constant_area[3]]

                if grayscale:
                    # keep the image grayscale
                    gray_vals = new_img.data.mean(1)
                    new_img.data = torch.stack([gray_vals] * 3, 1)

            
            self.acts.clear()
            self.grads.clear()

        # avoid side effects
        back_hook.remove()
        forw_hook.remove()

        return new_img, loss
