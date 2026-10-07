import numpy as np
import platform
import tempfile
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.transform import rescale
import math
import skimage as sk
from scipy import ndimage
from scipy import signal
from skimage import io


im=io.imread('mona-lisa.jpeg')
im2=im.copy()

n,m=im.shape[:2]
print(n,m)
omega=np.ones((n,m))
omega[20:40,20:40]=0
c=0
for i in range(n):
    for j in range(m):
        if omega[i,j]==0:
            c=c+1
print(c)

def trouverfrontiere(omega):
    n,m=omega.shape
    frontiere=[]
    for i in range(n):
        for j in range(m):
            if omega[i,j]==0:
                bord=False
                if i>0 and omega[i-1,j]==1:
                    bord=True
                if i<n-1 and omega[i+1,j]==1:
                    bord=True
                if j>0 and omega[i,j-1]==1:
                    bord=True
                if j<m-1 and omega[i,j+1]==1:
                    bord=True
                if bord:
                    frontiere.append((i,j))
    return frontiere

print(len(trouverfrontiere(omega)))


def choisir_patch(omega,frontiere):
    n,m=omega.shape
    h=1
    meilleur=frontiere[0]
    max_connus=-1
    for (i,j) in frontiere:
        connus=0
        for a in range(i-h,i+h+1):
            for b in range(j-h,j+h+1):
                if a>=0 and a<n and b>=0 and b<m:
                    if omega[a,b]==1:
                        connus=connus+1
        if connus>max_connus:
            max_connus=connus
            meilleur=(i,j)
    return meilleur

def remplacer_pixel(im,omega,i,j):
    n,m=omega.shape
    h=1
    dist_min=1000
    meilleur=(i,j)
    for a in range(i-h,i+h+1):
        for b in range(j-h,j+h+1):
            if a>=0 and a<n and b>=0 and b<m:
                if omega[a,b]==1:
                    d=(a-i)**2+(b-j)**2
                    if d<dist_min:
                        dist_min=d
                        meilleur=(a,b)
    im[i,j]=im[meilleur]
    omega[i,j]=1


im_trou=im.copy()
im_trou[omega==0]=255

while omega.sum()<n*m:
    frontiere=trouverfrontiere(omega)
    i,j=choisir_patch(omega,frontiere)
    remplacer_pixel(im2,omega,i,j)

plt.figure('Image originale')
plt.imshow(im)

plt.figure('Image masquée')
plt.imshow(im_trou)

plt.figure('Image reconstruite')
plt.imshow(im2)

plt.show()
