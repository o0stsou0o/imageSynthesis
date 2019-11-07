# Stimulus generation and general image optimization

This is some (at the moment very rough) code for optimizing images
based on neural network responses. Applications are visualizing the
preferences of single units (and population codes, once that's
implemented), and modifying stimuli by optimizing them to match
noisy versions of their intermediate representations, which might be
an interesting type of stimulus to generate for monkey experiments.

What's done so far is the core Optimizer class, which takes a given
network, layer, image and target response, and optimizes that image's
response at the layer to match the target (optionally keeping part of
the image constant). There are also a couple of wrapper functions, one
to visualize a given neuron's preference (starting from a random image),
and one to used a standard torchvision network (for now only Alexnet,
ResNet50, and VGG16) to generate modified versions of a folder full of
image stimuli.

Natural image regularization, in the form of alpha norm and total variation
norm (from the paper "Understanding Deep Image Representations by
Inverting Them"), has been added, but I'm not yet sure what the proper
weighting hyperparameters for it are yet. One major thing that still needs
to be added is more flexible wrapper functions.

A folder of example images from the 8000 dataset is included in the repository;
starting a python session, loading `wrappers.py` in, and running
`std_generate('alexnet', 5, '8k_images', 5)` will generate modified versions
of these stimuli based on noise (Gaussian with standard deviation 5)
added to the sixth layer of alexnet (where a layer is a convolution or
standard MLP layer). They will be saved, with identifying filenames,
in the `modified_8k_images` directory. These images are grayscale, but
the method works on color images as well (though the need for regularization
is more clear for those).
