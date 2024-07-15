# -*- coding: utf-8 -*-
"""
Digits recognition by physics principles

This program investigates if a digit classifier is possible by means
of physical investigations - in this context by the moment of inertia tensor in the
3d space axis directions
Created on Sun Nov 17 18:47:38 2019

"""
import cv2 
#import matplotlib
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
import pandas as pd
import sys


# images dict
samples = {}
J_dict = {}

# Some Constants
n_learn = 8 # number of learning samples per digit
n_samples = 11 # number of samples per digits in total

#______________________________________________________________________________
# Crop function (crops all white margins)
def crop(image, threshold = 255):
    """Crops any edges above or equal to threshold, crops blank image to 1x1.
    Returns cropped image.
    """
    # convert image[x,y]=[B,G,R] into image[x,y] = Max[B,G,R]
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2
    #crop white margins
    rows = np.where(np.min(flatImage, 0) < threshold)[0]
    if rows.size:
        cols = np.where(np.min(flatImage, 1) < threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image

#______________________________________________________________________________
# Binary Color, converts bgr val to 0 (white) and 1 (black)
def binaryC(bgr):
    if (bgr == [255,255,255]).all():
        return 0
    else:
        return 1
    
#______________________________________________________________________________
# read the sample pics, crop, normalize and save them in a dict 
for digit in range(9):
    for sample in range(n_samples):
        img_name = str(digit+1) + '_' + str(sample+1) # sample name
        img = cv2.imread('Ziffern/' + img_name + '.png') # load sample
        retval, threshold = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY) #convert to binary image
        crop_img = crop(threshold) # crop white margin
        width_nimg = int(crop_img.shape[1]/crop_img.shape[0]*70) # width of normalized img to height 70
        norm_img = cv2.resize(crop_img,(width_nimg,70)) # normalize to height 70, be careful - here x,y is assigned and x=j, y=i, so lines and cols change order !!!
        samples.update({img_name : norm_img}) # save sample in dict

#______________________________________________________________________________
# sample all images and calculate center of gravity and moment of inertia tensor
for name in samples.keys():
    J = np.zeros([3,3])
    img = samples[name]
    x_max = img.shape[0]
    y_max = img.shape[1]
    # calculate center of gravity (xs, ys) and total mass 
    xs = ys = mass = 0
    for x in range(0,x_max):
        for y in range(0,y_max):
            xs = xs + x*binaryC(img[x,y,0:3])
            ys = ys + y*binaryC(img[x,y,0:3])
            mass = mass + binaryC(img[x,y,0:3])
    xs = xs/mass
    ys = ys/mass
            
    # sum up J
    for x in range(0,x_max):
        for y in range(0,y_max):
            J[0,0] = J[0,0] + binaryC(img[x,y,0:3])*((y-ys)**2)
            J[0,1] = J[0,1] + binaryC(img[x,y,0:3])*(-(x-xs)*(y-ys))
            J[1,1] = J[1,1] + binaryC(img[x,y,0:3])*((x-xs)**2)
            J[2,2] = J[2,2] + binaryC(img[x,y,0:3])*((x-xs)**2 + (y-ys)**2)
    
    J[0,2] = J[2,0] = 0 
    J[1,2] = J[2,1] = 0
    J[1,0] = J[0,1]
    
    J = J/mass
    
    eigvalues, vectors = LA.eig(J)
    axis1 = eigvalues[0]*vectors[:,0]
    axis2 = eigvalues[1]*vectors[:,1]
    axis3 = eigvalues[2]*vectors[:,2]
    axis = np.hstack((axis1[0:2],axis2[0:2],axis3[2]))
    
    
    #J_dict.update({name : np.array([axis1[0:3],axis2[0:3],axis3[0:3]])}) # save J in dict
    J_dict.update({name : axis}) # save J in dict
    
#______________________________________________________________________________
# Plot result in 3d space
J_pd = pd.DataFrame.from_dict(J_dict)


#from mpl_toolkits.mplot3d import Axes3D
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#color = np.array(['b','g','r','c','m','y','k','w','b'])
#for n in range(1,10):
#    J1 = J_pd.iloc[:,((n-1)*11):(11*n)]
#    x = np.array(J1[0:1])
#    y = np.array(J1[1:2])
#    z = np.array(J1[2:3])
#    ax.scatter(x, y, z, c=color[n-1], marker='o')
#
#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')
#
#plt.show()
    
#______________________________________________________________________________
# Learn digits from 5d space clusters
data = J_pd.T # Transpose the J-Matrix
ind = data.index # take the digit descriptions
targets = [int(i[0]) for i in ind]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data,targets,test_size=0.50,random_state=0)


# k-neares-neighbor Modell aufstellen
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

# Predictions
y_predict=knn.predict(X_test)

# Erfolg beurteilen
print('\n Success of prediction: ',np.mean(y_predict==y_test))

#test statistiken
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test,y_predict))

# Überführung des Train-Datensatzes in Pandas und Matrix-Scatterplot
dataframe=pd.DataFrame(X_train)
scatterplot=pd.plotting.scatter_matrix(dataframe,c=y_train,marker='o')

        