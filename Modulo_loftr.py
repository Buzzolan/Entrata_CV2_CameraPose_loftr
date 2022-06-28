
"""
Created on Tue May 10 16:30:39 2022
this experiment shows how to use the loftr matcher with default config(dual-softmax) and the pretrained weights.
Work to do:
    1) Load model with 2D-3D correspondence;
    2) Extract array with correspondence;
    3) Call LoFTR with the main Image and another Image2;
    4) Filter the array of salient point in the main Image with the array of correspondence;
    5) Filter the array of salient point in the second Image with the filtered array of the main Image;
    6) Show the corrispondence between main image ande second image;
    7) Save the struct.
    
@author: tomma
"""

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

from mat4py import loadmat

import scipy.misc
import scipy.io

import json

#use a pre-trained model 
from src.loftr import LoFTR, default_cfg
from src.loftr.backbone import ResNetFPN_8_2, ResNetFPN_16_4

from matplotlib import image
import matplotlib.pyplot as plt
import open3d as o3d

#Carico path delle due immagini
img0_pth= "Enter_CV2_model/test/159/DSC_0159.JPG"
img1_pth= "Enter_CV2_model/test/159/DSC_0076.JPG"
fs1=0.18
fs2=0.18
indexImg='159'
indexTest='76'


'''Parte 1: Estrazione della mappa delle corrispondenze dall'immagine di riferimento
INPUT: file.m con matrice delle corrispondeze 2D-3D;
OUTPUT: array a due dimensioni contenente le coordinate pixel. Di queste coordiante pixel conosco il loro corrispettivo 
punto 3D '''

#load file.mat e conversione in array
path2D='models/Point2D_'+indexImg+'.mat'
path3D='models/Point3D_'+indexImg+'.mat'

data = loadmat(path2D)# tutti i punti 2D dell'immagne. Di questi punti 2D è noto il corrispettivo punto 3D
data3D = loadmat(path3D)# coordinate dei punti 3D

data_list=list(data.values())
data3D_list=list(data3D.values())
array_2D=[]
point_3d=[]
x=[]
y=[]
z=[]

for Dpoint in data3D_list[0]:
    point_3d.append(Dpoint)#creo lista dei punti 3D
    x=np.append(x,Dpoint[0])
    y=np.append(y,Dpoint[1])
    z=np.append(z,Dpoint[2])

#scatter dei punti 3D
# Creating figure
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
# Creating plot
ax.scatter3D(x, y, z, color = "green")
plt.title("proiezione dei punti 3D visibili dall'immagine di riferimento")
 
# show plot
plt.show()

'''
for point in data_list[0]:#creazione array 2D
    #applico fattore di scala per sottocampionamento=0.2
    point[0]=fs1*point[0]
    point[1]=fs1*point[1]     
    array_2D.append(point)'''
fig = plt.figure(figsize = (10, 7))   
RefImg=cv2.cvtColor(image.imread(img0_pth),cv2.COLOR_BGR2GRAY)


for coordinates in data_list[0]:
    x=coordinates[0]
    y=coordinates[1]
    
    plt.plot(x, y, color='red',marker='o',markersize=1)
    
plt.title("Punti sift con corrispondenza 3D")
plt.imshow(RefImg,cmap='Greys_r')
    
''' 
OUTPUT: -point_3d= punti 3D visibili dall'immagine di riferimento
        -data_list[0]= coordinate 2D (con punto 3D corrsipondente) dell'immagine di riferimento 
'''


''' Parte 2: Creazione della mappa delle corrispondenze 2D tra l'immagine di riferimento e
un'altra immagine presa da un punto di vista differente tramite LoFTR.
INPUT: Immagine di riferimento e Immagine scattata da un posto di vista differente
OUTPUT: Array con corrispondenze 2D tra le due immagini (ho le coordinate delle corrispondenze) '''
# The default config uses dual-softmax.
# You can change the default values like thr and coarse_match_type.
#load model
config=default_cfg #configurazione dell'algoritmo per LoFTR
matcher = LoFTR(config) #inizializzo il matcher e uso LoFTR
matcher.load_state_dict(torch.load("C:/Users/tomma/Documents/Università/Tesi/Codice_LoFTR/Loftr_Project/notebooks/weights/outdoor_ds.ckpt")['state_dict']) #position of my weights , carico i pesi del modello pre allenato
matcher = matcher.eval().cuda()



#Load two image and show it:
# one image is fixed because in the pipeline i have the 2D-3D correspondence



TestImg=cv2.cvtColor(image.imread(img1_pth),cv2.COLOR_BGR2GRAY)




SubRef_img = cv2.resize(RefImg, None, fx=fs1, fy=fs1, interpolation = cv2.INTER_CUBIC)
SubTest_img = cv2.resize(TestImg, None, fx=fs2, fy=fs2, interpolation = cv2.INTER_CUBIC)

SubRef_img0=cv2.resize(SubRef_img, (SubRef_img.shape[1]//8*8, SubRef_img.shape[0]//8*8)) 
SubTest_img1=cv2.resize(SubTest_img, (SubTest_img.shape[1]//8*8, SubTest_img.shape[0]//8*8))

img0_raw = cv2.imread(img0_pth, 0)
img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)

fig = plt.figure(figsize=(20, 14))

fig.add_subplot(1, 2, 1)
plt.imshow(SubRef_img0,cmap='Greys_r')
plt.axis('off')
plt.title("Immagine di riferimento")


fig.add_subplot(1, 2, 2)
plt.imshow(SubTest_img1,cmap='Greys_r')
plt.axis('off')
plt.title("Immagine di Test")


#imageRotate = cv2.rotate(img1_raw, cv2.cv2.ROTATE_90_CLOCKWISE)
#plt.imshow(imageRotate) 

#create the tensor structure from array and return a copy of this object in cuda memory
#img0 = torch.from_numpy(img0_raw)[None][None].cuda() /255.#
#img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
#batch = {'image0': img0, 'image1': img1}

#create the tensor structure from array and return a copy of this object in cuda memory
Ref0 = torch.from_numpy(SubRef_img0)[None][None].cuda() /255.#
Test1 = torch.from_numpy(SubTest_img1)[None][None].cuda() / 255.
batch = {'image0': Ref0, 'image1': Test1}



#Calcolo modulo LoFTR
#torch.cuda.empty_cache()
# Inference with LoFTR and get prediction
with torch.no_grad():# is lika a loop where every tensor inside the loop have reguires_grad set to False. Qui non mi serve perchè rete già allenata 
    matcher(batch)# al mio modello passo il batch, We test LoFTR’s speed on 100 randomly sampled ScanNet image pairs, with a size of 640 x 480.
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    expec= batch['expec_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()
    
i=0
best_mconf=[]
best_mkpts0=[]
best_mkpts1=[]
for value in mconf:
    if value > 0.4:
        best_mconf.append(value)
        best_mkpts0.append(mkpts0[i])
        best_mkpts1.append(mkpts1[i])
        
    i=i+1

best_mconf=np.asarray(best_mconf)
best_mkpts0=np.asarray(best_mkpts0)
best_mkpts1=np.asarray(best_mkpts1)
# Draw
color = cm.jet(best_mconf)
text = [
    'LoFTR',
    'Matches: {}'.format(len(best_mkpts0)),
]
fig = make_matching_figure(SubRef_img0,SubTest_img1, best_mkpts0, best_mkpts1, color, text=text)

#loftr ha inverito i punti?
fig = plt.figure(figsize=(20, 14))
best_loftr0=[]
for coordinates in best_mkpts0:
    x=coordinates[0]
    y=coordinates[1]
    best_loftr0.append([x,y])
    plt.plot(x, y, color='red',marker='o',markersize=1)
    
plt.title("Punti salienti trovati con loftr i quali hanno un suo corrispettivo nell'immagine di Test")
plt.imshow(SubRef_img0,cmap='Greys_r')

'''Parte 3): Filtro l'array delle corrsipondeze 2D tra le due immagini con l'array delle corrispondeze 2D-3D. Così da ottenere
solo le corrispondenze 2D tra le due immagini che hanno un loro corrispettivo punto 3D.
INPUT: Array 2D-3D e array corrsipondenze 2D;
OUTPUT; Arrayfiltrato con solo corrispondene 2D che hanno un corrispettivo punto 3D '''

#Devo ricordarmi che ho sottocampionato l'immagine prima di utilizzare loftr, mi conviene fare la ricerca nelle immagini sottocampionate così 
# è meno oneroso anche se poco meno accurato, allora lavoro solo con punti sottocampionati

int_2D_3D_0=[]
fig = plt.figure(figsize=(20, 14))

for coordinates in data_list[0]:
    x=coordinates[0]*fs1
    y=coordinates[1]*fs1
    int_2D_3D_0.append([x,y])
    plt.plot(x, y, color='red',marker='o',markersize=1)
    
plt.title("Punti proiettati correttamente nell'immagine originale")
plt.imshow(SubRef_img0,cmap='Greys_r')

#per fare il filtraggio ho bisogno di avere una lista e arrotondo i punti in interi

best_loftr0=np.asarray(best_loftr0).astype(int).tolist()
int_2D_3D_0=np.asarray(int_2D_3D_0).astype(int).tolist()#punti 2D con corrispettivo punto 3D
filtered_array1=[]#struttura finale con punti 2D che hanno corrispettivo 3D
filtered_point_3D=[]# punti 3D visibili dall'immagine di test
filtered_array0=[]#punti 2D trovati con loftr che hanno un corrispettivo 3D

fig = plt.figure(figsize=(20, 14))

for coordinates in best_loftr0:
    x=coordinates[0]
    y=coordinates[1]
    
    plt.plot(x, y, color='red',marker='o',markersize=1)
    
for coordinates in int_2D_3D_0:
    x=coordinates[0]
    y=coordinates[1]
    plt.plot(x, y, color='blue',marker='o',markersize=1)
    
plt.title("Punti proiettati correttamente nell'immagine originale")
plt.imshow(SubRef_img0,cmap='Greys_r')
c=0
for point in int_2D_3D_0:#per ogni punto nell'array con i punti 2D di cui conosco i corrispettivi punti 3D
# cerco in un range di 5 pixel se ho trovato un match con loftr 
    
#passo 1) ciclo per cercare corrispondenza in una finestra 3x3
    i=-1
    window=[0,0]#valore della finestra all'inizio della ricerca
    point_chek=False
    for x in range(5):
           #print(x)
           window[0]=point[0]+i+x
           
           for y in range(5):
              
               window[1]=point[1]+i+y
               #print("ora la finestra completa vale: ", window, "\n")
               #print("x e y sono:", x,y)
               #controllo se in questo pixel ho una corrispondenza
               #print(window)
               if window in best_loftr0:
                   
                   index=best_loftr0.index(window)
                   print("punto trovato\n")
                   c=c+1
                   print("il punto 2D con il corrispettivo punto 3D ha coordinate x= ",point[0]," e y= ",point[1])
                   print("il punto 2D trovato da loftr ha coordinate x= ",best_loftr0[index][0]," e y= ",best_loftr0[index][1])
                   
                   filtered_array1.append(best_mkpts1[index])
                   index3d=int_2D_3D_0.index(point)
                   filtered_point_3D.append(point_3d[index3d])
                   #print("trovato punto: ", point)
                   filtered_array0.append(best_loftr0[index])
                   point_chek=True
                   #print("esco dal primo for dato che chek: ", point_chek)
                   break
               
           if point_chek:
               #print("esco dal secono for")
               break
    



fig = plt.figure(figsize=(20, 14))
row=2
columns=2
fig.add_subplot(row,columns,1)

for coordinates in filtered_array0:
    x=coordinates[0]
    y=coordinates[1] #giro asse y
    plt.plot(x, y, color='white',marker='o',markersize=1)
    
plt.imshow(SubRef_img0)   
plt.title("ImgRef: Corrispondenze 2D trovate con LoFTR (conosco anche i corrispettivi 3D)")

fig.add_subplot(row,columns,2)

for coordinates in int_2D_3D_0:
    x=coordinates[0]
    y=coordinates[1] #giro asse y
    
    plt.plot(x, y, color='red',marker='o',markersize=1)
    
plt.imshow(SubRef_img0)
plt.title("ImgRef: Tutti i punti 2D che hanno una corrispondenza con un punto 3D")

fig.add_subplot(row,columns,3)

for coordinates in best_loftr0:
    x=coordinates[0]
    y=coordinates[1]
    
    plt.plot(x, y, color='blue',marker='o',markersize=1)
    
plt.imshow(SubRef_img0)
plt.title("ImgRef: Tutti i punti 2D trovati da LoFTR che hanno una corrsipondeza con l'immagine di test(non so il 3D)")

fig.add_subplot(row,columns,4)

for coordinates in filtered_array1:
    x=coordinates[0]
    y=coordinates[1]
    
    plt.plot(x, y, color='blue',marker='o',markersize=1)
    
plt.imshow(SubTest_img1)
plt.title("ImgTest: Punti 2D trovati con LoFTR dei quali conosco anche i corrispettivi punti 3D")

#punti 3D osservati dall'immagine di test
fig = plt.figure(figsize=(20, 14))
x=[]
y=[]
z=[]
for Dpoint in filtered_point_3D:
    x=np.append(x,Dpoint[0])
    y=np.append(y,Dpoint[1])
    z=np.append(z,Dpoint[2])
    

#scatter dei punti 3D
ax = plt.axes(projection ="3d")
ax.scatter3D(x, y, z, color = "green")

# Creating plot

plt.title("proiezione dei punti 3D visibili sia da Test che da Ref ma dei quali ho la corrsipondenza trovata con loftr")
 
# show plot
plt.show()

''' Mostro visivamente i nuovi match definitivi tra le due immagini''' 

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np

fig = plt.figure(figsize=(20,14))

ax1=fig.add_subplot(1,2,1)
for coordinates in filtered_array0:
    n=coordinates[0]
    z=coordinates[1]
    
    plt.plot(n, z, color='blue',marker='o',markersize=1)
    
plt.imshow(SubRef_img0)

ax2=fig.add_subplot(1,2,2)

for coordinates in filtered_array1:
    n=coordinates[0]
    z=coordinates[1]
    
    plt.plot(n, z, color='blue',marker='o',markersize=1)

for i in range(len(filtered_array1)):
    con = ConnectionPatch(xyA=filtered_array0[i][:], xyB=filtered_array1[i][:], coordsA="data", coordsB="data",
                          axesA=ax1, axesB=ax2, color="red")
    ax2.add_artist(con)
    
plt.imshow(SubTest_img1)


''' Creazione Output Finale. Essendo che inizialmente ho sottocampionato, ora vado a moltiplicare le coordinate pixel
    per l'inverso del fattore di scala così da poter utilizzare la stessa matrice K dei parametri interni. Filtered_arra1=contiene i punti 2D della 2nd Immagine presa da un pow differente dall'immagine di Ref che hanno una corrispondenza 2D trovate da loftr e i punti 3D
    3D_array= i corrispettivi punti 3D di Filtered_array1  '''

img1_final = image.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
fig = plt.figure(figsize=(20,14))
Up_filtered_array1 = []
Up_filtered_array0 = []
#per foto storte...
#reverse1=img1_final.shape[1]
#Up_filtered_array1_storto=[]

for coordinates in filtered_array1:
    n=coordinates[0]/fs2
    z=coordinates[1]/fs2
    Up_filtered_array1.append([n,z])
    plt.plot(n, z, color='blue',marker='o',markersize=1)


plt.imshow(img1_final)

for coordinates in filtered_array0:
    n=coordinates[0]/fs2
    z=coordinates[1]/fs2
    Up_filtered_array0.append([n,z])

Dict_corrispondenze_test = {"loftr_2D_checkTest": Up_filtered_array1, "loftr_3D_checkTest": filtered_point_3D}
Dict_corrispondenze_Ref = {"loftr_2D_checkRef": Up_filtered_array0, "loftr_3D_checkRef": filtered_point_3D}

scipy.io.savemat("Loftr_points"+indexTest+".mat", Dict_corrispondenze_test)
scipy.io.savemat("Loftr_points"+indexImg+".mat", Dict_corrispondenze_test)
