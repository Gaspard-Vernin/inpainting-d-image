from skimage import io as skio
import itertools
import math
import numpy as np

img = skio.imread("image.png").astype(float)

#on suppose que pixels_frontières comprend les positions des pixels a la frontiere
#pour l'instant on dit qu'on veut dupprimer le masque suivant : 
forme = list(itertools.product((10,11,12,13,14,15),(100,101,102,103,104)))
forme_set = set(forme) 
#on calcule la frontiere, un pixel qui est sur la frontière est dans la forme
#et a un voisin qui n'y est pas
frontiere = [] 
shape = img.shape
mouvements_simples = [(1,0),(0,1),(-1,0),(0,-1)]
for (i,j) in forme : 
    for m in mouvements_simples : 
        ni, nj = i + m[0], j + m[1]
        if (ni, nj) not in forme_set and 0 <= ni < shape[0] and 0 <= nj < shape[1]: 
            frontiere.append((i,j))
            break
#maintenant, on calcule le gradient de l'image avec l'orthogonal en chaque point
grads=[]
for (i,j) in frontiere:
    
    i_next = i + 1 if i + 1 < shape[0] else i
    j_next = j + 1 if j + 1 < shape[1] else j
    
    gradi = img[i_next, j, 0] - img[i, j, 0]
    gradj = img[i, j_next, 0] - img[i, j, 0]
    
    #on le tourne de 90 degré ce qui revient a faire 
    grad_tangent = (-gradj, gradi)
    grads.append(grad_tangent)

#maintenant on calcule les vecteurs n pour chaque point sur la frontiere 
n_vals=[]
for (i,j) in frontiere:
    point_droite = 1 if (i,j+1) in forme_set else 0 
    point_gauche = 1 if (i,j-1) in forme_set else 0 
    point_haut = 1 if (i+1,j) in forme_set else 0 
    point_bas = 1 if (i-1,j) in forme_set else 0 

    gradi = point_haut-point_bas 
    gradj = point_droite-point_gauche 
    norme = math.sqrt(gradi**2 + gradj**2)
    if norme == 0 : 
        gradi=0
        gradj=0
        print("??? norme == 0")
    else:
        # FIX : le vecteur normal doit être unitaire[cite: 1]
        gradi /= norme
        gradj /= norme
    n_vals.append((gradi,gradj)) # FIX : typo append

#maintenant on peut calculer D 
alpha = 255
C = [1] * len(frontiere) # FIX : initialisation de C pour que la boucle tourne

# FIX : vraie syntaxe de liste en compréhension avec variable d'itération 'k'
D_vals = [(frontiere[k], C[k]*np.abs(np.dot(grads[k],n_vals[k])/alpha)) for k in range(len(frontiere))]

# FIX : récupération du max via la fonction Python max sur le 2ème élément du tuple
best_pixel = max(D_vals, key=lambda x: x[1])[0]