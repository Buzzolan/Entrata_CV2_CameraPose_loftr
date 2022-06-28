# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 11:24:09 2022
Not finished yet

@author: tomma
"""

import os
os.chdir("..")# change working directory
from copy import deepcopy

import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils

from torch.autograd import Variable
import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure

import scipy.misc

import json

#use a pre-trained model 
from src.loftr import LoFTR, default_cfg
from src.loftr.backbone import ResNetFPN_8_2, ResNetFPN_16_4

# The default config uses dual-softmax.
# You can change the default values like thr and coarse_match_type.
#load model
config=default_cfg
matcher = LoFTR(config) #inizializzo il matcher e uso LoFTR
matcher.load_state_dict(torch.load("C:/Users/tomma/Documents/Università/Tesi/Codice_LoFTR/Loftr_Project/notebooks/weights/outdoor_ds.ckpt")['state_dict']) #position of my weights , carico modello pre allenato
matcher = matcher.eval().cuda()

#backbone = ResNetFPN_8_2(config['resnetfpn'])
#print(backbone)
#print(matcher)#info sulla rete
# A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor. Note that only layers with learnable parameters (convolutional layers, linear layers, etc.) and registered buffers (batchnorm’s running_mean) have entries in the model’s state_dict.
#print(default_cfg['coarse']) #configuration file, show layer in Tranform

#estrazione dei Conv2d layers:

#I will save the conv layers weights in this list
model_ResNet_weights=[]
# I will save the conv layer in this list
conv_layers=[]

#get all the model children as list
model_children = list(matcher.children())
model_children_resNet = list(model_children[0].children())

#counter to keep count of the conv layers
counter=0

#append all the conv layers and their respective wights to the list
for i in range(len(model_children_resNet)):
    if type(model_children_resNet[i]) == nn.Conv2d:
        counter+=1
        model_ResNet_weights.append(model_children_resNet[i].weight)
        conv_layers.append(model_children_resNet[i])
    elif type(model_children_resNet[i]) == nn.Sequential:
        for j in range(len(model_children_resNet[i])):
            #print(f"Verifico elemento: {j}")
            for child in model_children_resNet[i][j].children():
                #print(f"Child trovato: {child}")
                if type(child) == nn.Conv2d:
                    counter+=1
                    model_ResNet_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolution layers: {counter}")
print(conv_layers)


#Load two image and show it:
# one image is fixed because in the pipeline i have the 2D-3D correspondence
import matplotlib.pyplot as plt
from PIL import Image

fig = plt.figure(figsize=(10, 7))

img0_pth= "experiments/_SAM1020.jpg"
img1_pth= "experiments/test_1.jpg"
img0_raw = cv2.imread(img0_pth, 0)
img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)

#make a subsampling for reduce the batch size for GPU
img0_raw_sub = cv2.resize(img0_raw, None, fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)
img1_raw_sub = cv2.resize(img1_raw, None, fx=0.4, fy=0.4, interpolation = cv2.INTER_CUBIC)

#test image size for a fast computation in the network and the new image size shuold be divisible by 8
img0_raw = cv2.resize(img0_raw_sub, (img0_raw_sub.shape[1]//8*8, img0_raw_sub.shape[0]//8*8)) 
img1_raw = cv2.resize(img1_raw_sub, (img1_raw_sub.shape[1]//8*8, img1_raw_sub.shape[0]//8*8))

fig.add_subplot(1, 2, 1)
# showing image
plt.imshow(img0_raw)
plt.axis('off')
plt.title("First")

fig.add_subplot(1, 2, 2)
# showing image
plt.imshow(img1_raw)
plt.axis('off')
plt.title("Second")



#create the tensor structure from array and return a copy of this object in cuda memory
img0 = torch.from_numpy(img0_raw)[None][None].cuda() /255.#
img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
batch = {'image0': img0, 'image1': img1}

#Process image to every layer and append output and name of the layer to outputs[] and names[] lists


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0., std=1.)
])

image = Image.open(str('experiments/_SAM1020.jpg'))
plt.imshow(image)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image = transform(image)
print(f"Image shape before: {image.shape}")
image = image.unsqueeze(0)
print(f"Image shape after: {image.shape}")
image = image.to(device)

outputs = []
names = []
for layer in conv_layers[0:]:
    batch  = layer(batch )
    outputs.append(batch)
    names.append(str(layer))
print(len(outputs))#print feature_maps
for feature_map in outputs:
    print(feature_map.shape)
