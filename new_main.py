from skimage import io as skio
import itertools
import math
import numpy as np
from scipy.signal import correlate

img = skio.imread("../image2.png").astype(float)
if img.ndim == 2:            
    img = img[:, :, None]
shape = img.shape
height, width, nb_canaux = shape

#pour l'instant on dit qu'on veut dupprimer le masque suivant : 
taille_carre = 30
i0 = height // 2 - taille_carre // 2 
j0 = width // 2 - taille_carre // 2
forme = list(itertools.product(range(i0, i0 + taille_carre), range(j0, j0 + taille_carre)))
forme_origine = set(forme) #a check pk mettre un set ici ?
forme_set = set(forme)

taille_patch = 5
h = (taille_patch-1)//2
alpha = 255
mouvements_simples = [(1,0),(0,1),(-1,0),(0,-1)]
img_trou = img.copy()
for (i,j) in forme:
    img_trou[i, j, :] = 0
skio.imsave("image_trouee.png", img_trou.astype(np.uint8))

#on met l'image en niveaux de gris (plus tard on pourra le faire sur des couleurs mais c plus simple pr mtn)
gris = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        gris[i, j]=0
        for c in range(nb_canaux):
            gris[i, j] += img[i, j, c]
        gris[i, j] /= nb_canaux

#on remplit la matrice de confiance
confiance = np.ones((height, width))
for (i,j) in forme:
    confiance[i, j] = 0

#on presave tout les patchs 
sources = []
for i in range(h,height-h):
    for j in range(h,width-h):
        ok = True
        for k in range(-h,h+1):
            for l in range(-h,h+1):
                if (i+k,j+l) in forme_origine:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            sources.append((i,j))
print("on a",len(sources),"patchs")

def sobel(matrice, i, j):
    if not (1 <= i < matrice.shape[0] - 1 and 1 <= j < matrice.shape[1] - 1):
        return 0.0, 0.0
        
    gradj = (matrice[i-1, j+1] + 2*matrice[i, j+1] + matrice[i+1, j+1]) - (matrice[i-1, j-1] + 2*matrice[i, j-1] + matrice[i+1, j-1])
            
    gradi = (matrice[i+1, j-1] + 2*matrice[i+1, j] + matrice[i+1, j+1]) - (matrice[i-1, j-1] + 2*matrice[i-1, j] + matrice[i-1, j+1])
            
    return gradi, gradj
#tant qu'on a encore des pixels pas remplis
while len(forme_set) > 0:

    #calcul de la frontiere
    frontiere = []
    for (i,j) in forme_set:
        for m in mouvements_simples : 
            ni, nj = i + m[0], j + m[1]
            if (ni, nj) not in forme_set and 0 <= ni < height and 0 <= nj < width: 
                frontiere.append((i,j))
                break

    #calcul de I
    grads = []
    for (i,j) in frontiere:
        norme_max = -1
        gradi_max = 0
        gradj_max = 0
        for di in range(-h, h + 1):
            for dj in range(-h, h + 1):
                a, b = i + di, j + dj
                #on a besoin de (a,b), (a+1,b) et (a,b+1) connus et dans l'image
                if 0 <= a < height - 1 and 0 <= b < width - 1 \
                   and (a,b) not in forme_set and (a+1,b) not in forme_set and (a,b+1) not in forme_set:
                    gradi, gradj = sobel(gris, a, b)
                    norme = math.sqrt(gradi**2 + gradj**2)
                    if norme > norme_max:
                        norme_max = norme
                        gradi_max = gradi
                        gradj_max = gradj
        #on le tourne de 90 degré ce qui revient a faire 
        grad_tangent = (-gradj_max, gradi_max)
        grads.append(grad_tangent)

    #calcul de n
    n_vals = []
    for (i, j) in frontiere:
        gradi_n, gradj_n = sobel(confiance, i, j)
        norme = math.sqrt(gradi_n**2 + gradj_n**2)
        if norme == 0:
            n_vals.append((0.0, 0.0))
        else:
            n_vals.append((gradi_n / norme, gradj_n / norme))

    #calcul de la confiance 
    cval = []
    for (i,j) in frontiere:
        somme = 0
        aire = 0
        for k in range(-h, h + 1):
            for l in range(-h, h + 1):
                a, b = i + k, j + l
                if 0 <= a < height and 0 <= b < width:
                    aire += 1
                    somme += confiance[a, b]
        cval.append(somme / aire)

    meilleure_priorite = -1
    p_chapeau = None
    C_chapeau = 0
    for i in range(len(frontiere)):
        produit_scalair = abs(np.dot(grads[i], n_vals[i])/alpha)
        prio = cval[i] * (produit_scalair + 0.001)
        if prio > meilleure_priorite:
            meilleure_priorite = prio
            p_chapeau = frontiere[i]
            C_chapeau = cval[i]
    i_prio, j_prio = p_chapeau

    zone_inconnue = confiance[i_prio-h : i_prio+h+1, j_prio-h : j_prio+h+1].copy()
    patch_a_changer = np.zeros((taille_patch, taille_patch, nb_canaux))

    for c in range(nb_canaux):
        #on ne change que les pixels qu'on connait pas donc ceux ou zone inconnue=1
        patch_a_changer[:, :, c] = img[i_prio - h : i_prio + h + 1, j_prio - h : j_prio + h + 1, c] * zone_inconnue

    term3 = np.sum(patch_a_changer**2)

    ssd_map = np.zeros((height, width))
    for c in range(nb_canaux):
        im_canal_c = img[:, :, c]
        term1 = correlate(im_canal_c**2, zone_inconnue, mode='same', method='fft')
        term2 = -2 * correlate(im_canal_c, patch_a_changer[:, :, c], mode='same', method='fft')
        ssd_map += term1 + term2

    ssd_map += term3

    meilleur_ssd = float('inf')
    q = None
    for (i, j) in sources:
        ssd = round(ssd_map[i, j], 2)
        if ssd < meilleur_ssd:
            meilleur_ssd = ssd
            q = (i, j)
            
    for i in range(-h, h + 1):
        for j in range(-h, h + 1):
            a, b = i_prio + i, j_prio + j
            if 0 <= a < height and 0 <= b < width and (a,b) in forme_set:
                for c in range(nb_canaux):
                    img[a, b, c] = img[q[0] + i, q[1] + j, c]
                s = 0
                for c in range(nb_canaux):
                    s += img[a, b, c]
                gris[a, b] = s / nb_canaux
                confiance[a, b] = C_chapeau
                forme_set.remove((a,b))

    print("pixels pas encore remplis", len(forme_set))

skio.imsave("resultat.png", np.clip(img, 0, 255).astype(np.uint8))
print("résultat enregistré dans resultat.png")