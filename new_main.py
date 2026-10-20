import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
import math
from scipy.signal import correlate

def get_mask_interactively(image_path, brush_size=15):
    img_raw = skio.imread(image_path)
    
    img_display = img_raw.astype(np.float32)
    if img_display.max() > 0:
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
        
    # Gestion des niveaux de gris et canal alpha pour l'affichage
    if img_display.ndim == 2:
        img_display = np.stack((img_display,)*3, axis=-1)
    else:
        img_display = img_display[:, :, :3]
        
    H, W = img_display.shape[:2]
    omega = np.zeros((H, W), dtype=bool)
    yy, xx = np.ogrid[:H, :W]         
    state = {"drawing": False, "erase": False}

    fig, ax = plt.subplots()
    ax.imshow(img_display, interpolation="none")
    overlay = ax.imshow(np.where(omega, 1.0, np.nan), alpha=0.5, cmap="Reds", vmin=0, vmax=1)
    ax.set_title("Clic GAUC HE: Dessiner | Clic DROIT: Effacer\nFermez la fenêtre pour valider")

    def paint(event):
        if event.inaxes != ax or not state["drawing"]:
            return
        x, y = int(event.xdata), int(event.ydata)
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= brush_size ** 2
        omega[disk] = not state["erase"]
        overlay.set_data(np.where(omega, 1.0, np.nan))
        fig.canvas.draw_idle()

    def on_press(event):
        state["drawing"] = True
        state["erase"] = (event.button == 3)
        paint(event)

    def on_release(event):
        state["drawing"] = False

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", paint)

    plt.show()
    
    coords = np.argwhere(omega)
    return [(int(c[0]), int(c[1])) for c in coords]



chemin="../im.png"

forme = get_mask_interactively(chemin, brush_size=26)


img = skio.imread(chemin).astype(float)
if img.ndim == 2:            
    img = img[:, :, None]
img = img[:, :, :3]

shape = img.shape
height, width, nb_canaux = shape

forme_origine = set(forme) 
forme_set = set(forme)
print(forme_origine)
taille_patch = 17
h = (taille_patch - 1) // 2
alpha = 255
mouvements_simples = [(1,0),(0,1),(-1,0),(0,-1)]

img_trou = img.copy()
for (i,j) in forme:
    img_trou[i, j, :] = 0
skio.imsave("capture.png", img_trou.astype(np.uint8))

#on met l'image en niveaux de gris (plus tard on pourra le faire sur des couleurs mais c plus simple pr mtn)
gris = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        gris[i, j]=0
        for c in range(nb_canaux):
            gris[i, j] += img[i, j, c]
        gris[i, j] /= nb_canaux

#on remplit la mask de confiance
confiance = np.ones((height, width))
for (i,j) in forme:
    confiance[i, j] = 0

#on presave tout les patchs 
sources = []
print("ok")
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

def sobel(m, i, j):
    if not (1 <= i < m.shape[0] - 1 and 1 <= j < m.shape[1] - 1):
        return 0.0, 0.0
 
    gradj = (m[i-1, j+1]+2*m[i, j+1]+m[i+1, j+1]) - (m[i-1, j-1]+2*m[i, j-1]+m[i+1, j-1])
            
    gradi = (m[i+1, j-1]+2*m[i+1, j]+m[i+1, j+1]) - (m[i-1, j-1]+2*m[i-1, j]+m[i-1, j+1])
    gradj = m[i, j+1] - m[i, j-1]
    gradi = m[i+1, j] - m[i-1, j]        
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

    i_min, i_max = max(0, i_prio - h), min(height, i_prio + h + 1)
    j_min, j_max = max(0, j_prio - h), min(width, j_prio + h + 1)

    pi_min, pi_max = h - (i_prio - i_min), h + (i_max - i_prio)
    pj_min, pj_max = h - (j_prio - j_min), h + (j_max - j_prio)

 
    zone_inconnue = np.zeros((taille_patch, taille_patch))
    zone_inconnue[pi_min:pi_max, pj_min:pj_max] = confiance[i_min:i_max, j_min:j_max]

    patch_a_changer = np.zeros((taille_patch, taille_patch, nb_canaux))

    for c in range(nb_canaux):
 
        patch_a_changer[pi_min:pi_max, pj_min:pj_max, c] = img[i_min:i_max, j_min:j_max, c] * zone_inconnue[pi_min:pi_max, pj_min:pj_max]

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
#méthode itérative
vote = np.zeros_like(img)
poid = np.zeros_like(img)
s=6
img_trou_ech = [forme[i] for i in range(len(forme)) if (forme[i][0] % s == 0 and forme[i][1] % s == 0)]
print("gogo")
nb_i=2
mediane = True
temperature=1000
for k in range(nb_i):
    skio.imsave(f"resultat_{k}.png", np.clip(img, 0, 255).astype(np.uint8))
    votes_mediane = {}
    for (i, j) in img_trou_ech:
        i_min, i_max = max(0, i - h), min(height, i + h + 1)
        j_min, j_max = max(0, j - h), min(width, j + h + 1)
        
        patch = img[i_min:i_max, j_min:j_max].copy()
        mask = np.ones((i_max - i_min, j_max - j_min))
        ssd_map = np.zeros((height, width))
        term3 = np.sum(patch ** 2)
        
        for c in range(nb_canaux):
            im_canal = img[:, :, c]
            term1 = correlate(im_canal**2, mask, mode='same', method='fft')
            term2 = -2 * correlate(im_canal, patch[:, :, c], mode='same', method='fft')
            ssd_map += term1 + term2
        ssd_map += term3
        
        meilleur_ssd = float('inf')
        q = None
        temperature=0
        for (si, sj) in sources:
            ssd = ssd_map[si, sj]
            if ssd < meilleur_ssd:
                meilleur_ssd = ssd
                qi,qj = (si, sj) 
                temperature+=meilleur_ssd
            temperature/=len(sources)
            if mediane == False :
                di_haut, di_bas = i - i_min, i_max - i
                dj_gauche, dj_droite = j - j_min, j_max - j
                
                patch_source = img[qi - di_haut : qi + di_bas, qj - dj_gauche : qj + dj_droite]
                poid = np.exp(-meilleur_ssd/temperature)
                vote[i_min:i_max, j_min:j_max] += patch_source*poid
                poid[i_min:i_max, j_min:j_max] += poid
    
            else:  
                di_haut, di_bas = i - i_min, i_max - i
                dj_gauche, dj_droite = j - j_min, j_max - j
                
                patch_source = img[qi - di_haut : qi + di_bas, qj - dj_gauche : qj + dj_droite]
                
                for d_i in range(patch_source.shape[0]):
                    for d_j in range(patch_source.shape[1]):
                        pi, pj = i_min + d_i, j_min + d_j
                        if (pi, pj) not in votes_mediane:
                            votes_mediane[(pi, pj)] = []
                        votes_mediane[(pi, pj)].append(patch_source[d_i, d_j])
    
    if mediane == False:
        mask_poids = poid > 0
        img[mask_poids] = vote[mask_poids] / poid[mask_poids]
    else:
        for (pi, pj), liste_valeurs in votes_mediane.items():
            img[pi, pj] = np.median(liste_valeurs, axis=0)
    print("ca roule")
skio.imsave("resultat.png", np.clip(img, 0, 255).astype(np.uint8))
print("résultat enregistré dans resultat.png")


 