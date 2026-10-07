
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
im3=im.copy()
im4=im.copy()


n,m=im.shape[:2]
print(n,m)
omega=np.ones((n,m))
omega[50:120,100:180]=0
c=0
for i in range(n):
    for j in range(m):
        if omega[i,j]==0:
            c=c+1
print(c)

omega3=omega.copy()
omega4=omega.copy()


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



def choisirpatch(omega,frontiere):
    n,m=omega.shape
    h=5
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

def remplacerpixel(im,omega,i,j):
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
    i,j=choisirpatch(omega,frontiere)
    remplacerpixel(im2,omega,i,j)


def get_gau_ker(s):
    ss=int(max(3,2*np.round(2.5*s)+1))
    ms=(ss-1)//2
    X=np.arange(-ms,ms+0.99)
    y=np.exp(-X**2/2/s**2)
    out=y.reshape((ss,1))@y.reshape((1,ss))
    out=out/out.sum()
    return out

"""
def filtre_lineaire(im,mask):
    #renvoie la convolution de l'image avec le mask. Le calcul se fait en utilisant la transformee de Fourier et est donc circulaire.  Fonctionne seulement pour les images en niveau de gris.

    fft2=np.fft.fft2
    ifft2=np.fft.ifft2
    (y,x)=im.shape
    (ym,xm)=mask.shape
    mm=np.zeros((y,x))
    mm[:ym,:xm]=mask
    fout=(fft2(im)*fft2(mm))
    # on fait une translation pour ne pas avoir de decalage de l'image
    # pour un mask de taille impair ce sera parfait, sinon, il y a toujours un decalage de 1/2
    mm[:ym,:xm]=0
    y2=int(np.round(ym/2-0.5))
    x2=int(np.round(xm/2-0.5))
    mm[y2,x2]=1
    out=np.real(ifft2(fout*np.conj(fft2(mm))))
    return out

    
def filtre_lineaire_couleur(im,mask):
    out=np.zeros(im.shape)
    for c in range(im.shape[2]):
        out[:,:,c]=filtre_lineaire(im[:,:,c],mask)
    return out

"""

def remplacerpixelgauss(im,omega,i,j,s=1):
    n,m=omega.shape
    noyau=get_gau_ker(s)
    h=(noyau.shape[0]-1)//2
    somme=0
    total_poids=0
    for a in range(i-h,i+h+1):
        for b in range(j-h,j+h+1):
            if a>=0 and a<n and b>=0 and b<m:
                if omega[a,b]==1:
                    poids=noyau[a-i+h,b-j+h]
                    somme=somme+poids*im[a,b]
                    total_poids=total_poids+poids
    im[i,j]=somme/total_poids
    omega[i,j]=1

while omega3.sum()<n*m:
    frontiere=trouverfrontiere(omega3)
    i,j=choisirpatch(omega3,frontiere)
    remplacerpixelgauss(im3,omega3,i,j,s=1)

def remplacerpixelmoyenne(im,omega,i,j,h=1):
    n,m=omega.shape
    somme=0
    nb=0
    for a in range(i-h,i+h+1):
        for b in range(j-h,j+h+1):
            if a>=0 and a<n and b>=0 and b<m:
                if omega[a,b]==1:
                    somme=somme+im[a,b].astype(float)
                    nb=nb+1
    im[i,j]=somme/nb
    omega[i,j]=1


while omega4.sum()<n*m:
    frontiere=trouverfrontiere(omega4)
    i,j=choisirpatch(omega4,frontiere)
    remplacerpixelmoyenne(im4,omega4,i,j,h=1)

plt.figure('Image reconstruite moyenne gauss')
plt.imshow(im3)

plt.figure('Image reconstruite moyenne')
plt.imshow(im4)


plt.figure('Image reconstruite')
plt.imshow(im2)


plt.figure('Image masquée')
plt.imshow(im_trou)


plt.figure('Image originale')
plt.imshow(im)

plt.show()

