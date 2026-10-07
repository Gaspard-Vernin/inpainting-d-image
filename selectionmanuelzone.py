import numpy as np
import matplotlib.pyplot as plt

"""
i = plt.imread("mona-lisa.jpeg")


plt.imshow(i, interpolation="none")
plt.show()

H, W = i.shape[:2]

# Omega = masque booléen, True = zone à enlever
omega = np.zeros((H, W), dtype=bool)
omega[80:180, 120:200] = True   # à ajuster 

plt.imshow(i)
plt.imshow(omega, alpha=0.4, cmap="Reds")  # superpose le masque en rouge transparent
plt.title("Omega (zone à retirer)")
plt.show()
"""
def draw_mask(I, brush=4):
    H, W = I.shape[:2]
    omega = np.zeros((H, W), dtype=bool)
    yy, xx = np.ogrid[:H, :W]         
    state = {"drawing": False, "erase": False}

    fig, ax = plt.subplots()
    ax.imshow(I, interpolation="none")
    overlay = ax.imshow(np.where(omega, 1.0, np.nan), alpha=0.5, cmap="Reds", vmin=0, vmax=1)

    def paint(event):
        if event.inaxes != ax or not state["drawing"]:
            return
        x, y = int(event.xdata), int(event.ydata)
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= brush ** 2
        omega[disk] = not state["erase"]
        overlay.set_data(np.where(omega, 1.0, np.nan))
        fig.canvas.draw_idle()
        return(x,y)

    def on_press(event):
        state["drawing"] = True
        state["erase"] = (event.button == 3)   # 1 = gauche, 3 = droit
        paint(event)

    def on_release(event):
        state["drawing"] = False

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", paint)

    plt.show()      
    return omega


I = plt.imread("mona-lisa.jpeg")
n,m=I.shape[:2]


omega = draw_mask(I)

C = np.ones((n, m))
C[omega] = 0













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
    norme = math.sqrt(gradi*2 + gradj*2)
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
C = [1] * len(frontiere) 
D_vals = [(frontiere[k], C[k]*np.abs(np.dot(grads[k],n_vals[k])/alpha)) for k in range(len(frontiere))]
ixel = max(D_vals, key=lambda x:x[1])[0]

