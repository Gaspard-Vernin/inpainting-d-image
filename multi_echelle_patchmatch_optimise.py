import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage.color import rgb2lab, lab2rgb   
from skimage.transform import resize        
import math
from scipy.signal import correlate
from scipy import ndimage   

def vers_rgb_uint8(img_lab):
    rgb = lab2rgb(img_lab)
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)

def get_mask_interactively(image_path, brush_size=15):
    img_raw = skio.imread(image_path)

    img_display = img_raw.astype(np.float32)
    if img_display.max() > 0:
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())

    # Gestion des niveaux de luminance et canal alpha pour l'affichage
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
    ax.set_title("Clic GAUCHE: Dessiner | Clic DROIT: Effacer\nFermez la fenêtre pour valider")  # [CORRIGÉ] typo "GAUC HE"

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



chemin="../bast-pau.png"

forme = get_mask_interactively(chemin, brush_size=30)


img = skio.imread(chemin).astype(float)
if img.ndim == 2:
    img = img[:, :, None]
img = img[:, :, :3]

if img.shape[2] == 1:
    img = np.repeat(img, 3, axis=2)

shape = img.shape
height, width, nb_canaux = shape

forme_origine = set(forme)
forme_set = set(forme)

taille_patch = 13
h = (taille_patch - 1) // 2

alpha = 4 * 100

largeur_bande = 60
lambda_dist = 0.02

poids_guide = 0.3                 
taille_min = 6 * taille_patch     

img_trou = img.copy()
for (i,j) in forme:
    img_trou[i, j, :] = 0
skio.imsave("capture.png", img_trou.astype(np.uint8))

img = rgb2lab(img / 255.0)

trou = np.zeros((height, width), dtype=bool)
for (i,j) in forme:
    trou[i, j] = True



def preparer_sources(trou, largeur_bande):
    height, width = trou.shape
    valides = ndimage.binary_erosion(~trou, structure=np.ones((taille_patch, taille_patch)), border_value=0)

    masque_sources = valides
    if largeur_bande is not None:
        dist_au_trou = ndimage.distance_transform_edt(~trou)
        dans_la_bande = dist_au_trou <= largeur_bande
        if (valides & dans_la_bande).sum() >= 200:
            masque_sources = valides & dans_la_bande

    if masque_sources.sum() == 0:
        raise ValueError("Aucun patch source valide : le trou est trop grand pour taille_patch=%d" % taille_patch)

    a, b = np.nonzero(masque_sources)
    r0, r1 = max(0, a.min() - h), min(height, a.max() + h + 1)
    c0, c1 = max(0, b.min() - h), min(width, b.max() + h + 1)

    diag = math.hypot(height, width)         #normalisaion pr que la penalie lambda rese oujours la meme

    valides = np.argwhere(masque_sources)            
    valides_i, valides_j = valides[:, 0], valides[:, 1]
    return valides_i, valides_j, r0, r1, c0, c1, diag


def meilleure_source(ssd_map_c, nb_pixels_connus, i_p, j_p, src):
    valides_i, valides_j, r0, r1, c0, c1, diag = src
    nb_canaux = 3
    ssd = ssd_map_c[valides_i - r0, valides_j - c0 ]
    mse = ssd / (nb_pixels_connus * nb_canaux)
    dist = np.sqrt((valides_i - i_p) ** 2 + (valides_j - j_p) ** 2) / diag
    cout = np.round(mse + lambda_dist * 100 ** 2 * dist, 2)
    k = np.argmin(cout)
    return (int(valides_i[k]), int(valides_j[k])), max(float(mse[k]), 0.0)



def chercher_source(img, poids, cible, i_p, j_p, src, depart=None):   # [PATCHMATCH] paramètre depart ajouté (ignoré ici)
    r0, r1, c0, c1 = src[2:6]
    nb_canaux = img.shape[2]

    patch_a_changer = cible * poids[:, :, None]
    term3 = np.sum(patch_a_changer * cible)        # = somme des poids * valeur²

    img_zone = img[r0:r1, c0:c1]
    ssd_map = np.zeros((r1 - r0, c1 - c0))
    for c in range(nb_canaux):
        im_canal_c = img_zone[:, :, c]
        term1 = correlate(im_canal_c**2, poids, mode='same', method='fft')
        term2 = -2 * correlate(im_canal_c, patch_a_changer[:, :, c], mode='same', method='fft')
        ssd_map += term1 + term2
    ssd_map += term3

    return meilleure_source(ssd_map, poids.sum(), i_p, j_p, src)


# ==================== [PATCHMATCH] DEBUT DU BLOC AJOUTE ====================
# Même entrées / sortie que chercher_source, mais sans tester toutes les sources :
#   1. on part de quelques candidats : ceux de 'depart' (la "propagation") + quelques sources au hasard
#   2. recherche aléatoire autour du meilleur candidat, dans un rayon divisé par 2 à chaque essai
# depart = liste de centres sources (i, j) proposés par l'appelant : si un pixel voisin a trouvé
# une bonne source, la source décalée est sans doute bonne aussi.
nb_aleatoires = 30     # candidats tirés au hasard au départ
nb_essais = 16         # candidats testés à chaque rayon de la recherche aléatoire
nb_tours = 2           # nombre de recherches aléatoires complètes (du grand rayon au rayon 1)

def chercher_source_patchmatch(img, poids, cible, i_p, j_p, src, depart=None):
    valides_i, valides_j, r0, r1, c0, c1, diag = src
    height, width, nb_canaux = img.shape
    lin_valides = valides_i * width + valides_j      # déjà trié : argwhere parcourt l'image ligne par ligne
    d = np.arange(-h, h + 1)

    def est_valide(ci, cj):
        dedans = (ci >= 0) & (ci < height) & (cj >= 0) & (cj < width)
        lin = ci * width + cj
        pos = np.clip(np.searchsorted(lin_valides, lin), 0, len(lin_valides) - 1)
        return dedans & (lin_valides[pos] == lin)

    def couts(ci, cj):
        # tous les patchs candidats d'un coup : (nb candidats, taille_patch, taille_patch, nb_canaux)
        patchs = img[ci[:, None, None] + d[None, :, None], cj[:, None, None] + d[None, None, :]]
        ssd = np.sum(poids[None, :, :, None] * (patchs - cible[None]) ** 2, axis=(1, 2, 3))
        mse = ssd / (poids.sum() * nb_canaux)
        dist = np.sqrt((ci - i_p) ** 2 + (cj - j_p) ** 2) / diag
        return mse + lambda_dist * 100 ** 2 * dist, mse     # même coût que meilleure_source

    # 1. candidats de départ : au hasard + ceux fournis par l'appelant
    k = np.random.randint(0, len(valides_i), nb_aleatoires)
    ci, cj = valides_i[k], valides_j[k]
    if depart is not None and len(depart) > 0:
        depart = np.unique(np.asarray(depart, dtype=int).reshape(-1, 2), axis=0)
        ok = est_valide(depart[:, 0], depart[:, 1])
        ci = np.concatenate([ci, depart[ok, 0]])
        cj = np.concatenate([cj, depart[ok, 1]])
    c, m = couts(ci, cj)
    b = np.argmin(c)
    bi, bj, bc, bm = ci[b], cj[b], c[b], m[b]

    # 2. recherche aléatoire autour du meilleur, rayon divisé par 2 à chaque essai
    for tour in range(nb_tours):
        r = max(height, width)
        while r >= 1:
            ci = bi + np.random.randint(-r, r + 1, nb_essais)
            cj = bj + np.random.randint(-r, r + 1, nb_essais)
            ok = est_valide(ci, cj)
            if ok.any():
                c, m = couts(ci[ok], cj[ok])
                b = np.argmin(c)
                if c[b] < bc:
                    bi, bj, bc, bm = ci[ok][b], cj[ok][b], c[b], m[b]
            r = r // 2

    return (int(bi), int(bj)), max(float(bm), 0.0)


# LA ligne qui choisit la méthode de recherche (utilisée aux deux endroits où on cherche un patch) :
chercher = chercher_source_patchmatch      # PatchMatch (approchée, plus rapide)
# chercher = chercher_source               # recherche exhaustive par FFT (exacte, plus lente)
# ==================== [PATCHMATCH] FIN DU BLOC AJOUTE ====================


def sobel(m, i, j):
    if not (1 <= i < m.shape[0] - 1 and 1 <= j < m.shape[1] - 1):
        return 0.0, 0.0

    gradj = (m[i-1, j+1]+2*m[i, j+1]+m[i+1, j+1]) - (m[i-1, j-1]+2*m[i, j-1]+m[i+1, j-1])

    gradi = (m[i+1, j-1]+2*m[i+1, j]+m[i+1, j+1]) - (m[i-1, j-1]+2*m[i-1, j]+m[i-1, j+1])
    return gradi, gradj


def remplir(img, trou, guide=None, largeur_bande=None):
    img = img.copy()
    height, width, nb_canaux = img.shape

    forme = [(int(i), int(j)) for (i, j) in np.argwhere(trou)]
    forme_set = set(forme)

    luminance = img[:, :, 0].copy()

    #on remplit la mask de confiance
    confiance = np.ones((height, width))
    for (i,j) in forme:
        confiance[i, j] = 0

    connu = np.ones((height, width))
    for (i,j) in forme:
        connu[i, j] = 0

    reste = trou.copy()

    origine = np.full((height, width, 2), -1)   # [PATCHMATCH] pour chaque pixel rempli : le pixel source d'où il a été copié

    src = preparer_sources(trou, largeur_bande)
    r0, r1, c0, c1 = src[2:6]

    #tant qu'on a encore des pixels pas remplis
    while len(forme_set) > 0:

        croix = ndimage.generate_binary_structure(2, 1)
        masque_front = reste & ~ndimage.binary_erosion(reste, structure=croix, border_value=1)
        frontiere = [(int(i), int(j)) for (i, j) in np.argwhere(masque_front)]

        # ==================== [VECTORISE] DEBUT : I, n et priorité calculés sans boucle sur le front ====================
        # coordonnées du front en tableaux (même ordre que la liste frontiere)
        fi, fj = np.nonzero(masque_front)

        #calcul de I
        # [VECTORISE] sobel de toute l'image d'un coup, au lieu d'un appel par pixel
        gi = ndimage.sobel(luminance, axis=0)
        gj = ndimage.sobel(luminance, axis=1)
        # [VECTORISE] un pixel est utilisable si les 9 pixels de son voisinage sont connus et dans l'image
        # (remplace le test all((a+x, b+y) not in forme_set ...) et la condition 1 <= a < height-1)
        grad_ok = ndimage.binary_erosion(connu.astype(bool), structure=np.ones((3, 3)), border_value=0)
        norme_img = np.where(grad_ok, np.sqrt(gi**2 + gj**2), -1.0)      # -1 = inutilisable, comme ton norme_max de départ
        # [VECTORISE] pour chaque pixel du front : les taille_patch x taille_patch normes de son patch
        # (hors image -> -2, jamais choisi car le centre du patch vaut au moins -1)
        offs = np.arange(-h, h + 1)
        pad = np.pad(norme_img, h, constant_values=-2.0)
        fenetres = pad[(fi + h)[:, None, None] + offs[None, :, None], (fj + h)[:, None, None] + offs[None, None, :]]
        k_max = fenetres.reshape(len(fi), -1).argmax(axis=1)             # premier maximum, comme ton ">" strict
        a_max = fi + offs[k_max // taille_patch]
        b_max = fj + offs[k_max % taille_patch]
        valide = norme_img[a_max, b_max] >= 0                            # sinon aucun pixel utilisable : gradient (0, 0)
        gradi_max = np.where(valide, gi[a_max, b_max], 0.0)
        gradj_max = np.where(valide, gj[a_max, b_max], 0.0)
        #on le tourne de 90 degré ce qui revient a faire
        tang_i, tang_j = -gradj_max, gradi_max

        #calcul de n
        # [VECTORISE] sobel de toute l'image, lu aux positions du front
        ni = ndimage.sobel(connu, axis=0)[fi, fj]
        nj = ndimage.sobel(connu, axis=1)[fi, fj]
        bord = (fi < 1) | (fi >= height - 1) | (fj < 1) | (fj >= width - 1)   # ton sobel renvoie (0, 0) sur le bord de l'image
        ni = np.where(bord, 0.0, ni)
        nj = np.where(bord, 0.0, nj)
        norme_n = np.sqrt(ni**2 + nj**2)
        non_nul = norme_n > 0
        ni = np.where(non_nul, ni / np.where(non_nul, norme_n, 1.0), 0.0)
        nj = np.where(non_nul, nj / np.where(non_nul, norme_n, 1.0), 0.0)

        carte_conf = ndimage.uniform_filter(confiance, size=taille_patch, mode='constant', cval=0.0)
        cval = carte_conf[fi, fj]                                        # [VECTORISE]

        # [VECTORISE] priorité de tous les pixels du front d'un coup, puis argmax (premier maximum, comme avant)
        produit_scalair = np.abs((tang_i * ni + tang_j * nj) / alpha)
        prio = cval * (produit_scalair + 0.001)
        k_prio = np.argmax(prio)
        p_chapeau = frontiere[k_prio]
        C_chapeau = cval[k_prio]
        i_prio, j_prio = p_chapeau
        # ==================== [VECTORISE] FIN ====================

        i_min, i_max = max(0, i_prio - h), min(height, i_prio + h + 1)
        j_min, j_max = max(0, j_prio - h), min(width, j_prio + h + 1)

        pi_min, pi_max = h - (i_prio - i_min), h + (i_max - i_prio)
        pj_min, pj_max = h - (j_prio - j_min), h + (j_max - j_prio)



        zone_inconnue = np.zeros((taille_patch, taille_patch))
        zone_inconnue[pi_min:pi_max, pj_min:pj_max] = connu[i_min:i_max, j_min:j_max]

        valeurs = img[i_min:i_max, j_min:j_max].copy()
        if guide is not None:
            inconnus = (connu[i_min:i_max, j_min:j_max] == 0)
            valeurs[inconnus] = guide[i_min:i_max, j_min:j_max][inconnus]
            zone_inconnue[pi_min:pi_max, pj_min:pj_max] += poids_guide * inconnus

        cible = np.zeros((taille_patch, taille_patch, nb_canaux))
        cible[pi_min:pi_max, pj_min:pj_max] = valeurs

        # [PATCHMATCH] candidats de départ : chaque pixel déjà rempli du patch vient d'une source ;
        # cette source, décalée de la position du pixel dans le patch, donne un centre de patch source candidat
        zone_origine = origine[i_min:i_max, j_min:j_max].reshape(-1, 2)
        decal_i, decal_j = np.mgrid[i_min - i_prio:i_max - i_prio, j_min - j_prio:j_max - j_prio]
        rempli = zone_origine[:, 0] >= 0
        depart = zone_origine[rempli] - np.stack([decal_i.ravel(), decal_j.ravel()], axis=1)[rempli]

        q, _ = chercher(img, zone_inconnue, cible, i_prio, j_prio, src, depart)   # [PATCHMATCH] chercher_source -> chercher

        for i in range(-h, h + 1):
            for j in range(-h, h + 1):
                a, b = i_prio + i, j_prio + j
                if 0 <= a < height and 0 <= b < width and (a,b) in forme_set:
                    for c in range(nb_canaux):
                        img[a, b, c] = img[q[0] + i, q[1] + j, c]
                    origine[a, b] = (q[0] + i, q[1] + j)   # [PATCHMATCH] on retient d'où ce pixel a été copié

                    luminance[a, b] = img[a, b, 0]
                    confiance[a, b] = C_chapeau
                    connu[a, b] = 1
                    reste[a, b] = False
                    forme_set.remove((a,b))

        print("pixels pas encore remplis", len(forme_set))
    return img



def reduire(img, trou):
    connu = (~trou).astype(float)
    flou_connu = ndimage.gaussian_filter(connu, sigma=1.0)
    img_red = np.zeros_like(img)
    for c in range(img.shape[2]):
        img_red[:, :, c] = ndimage.gaussian_filter(img[:, :, c] * connu, sigma=1.0) / np.maximum(flou_connu, 1e-8)
    img_red = img_red[::2, ::2]

    H2, W2 = img_red.shape[:2]
    trou_pad = np.zeros((2 * H2, 2 * W2), dtype=bool)
    trou_pad[:trou.shape[0], :trou.shape[1]] = trou
    trou_red = trou_pad.reshape(H2, 2, W2, 2).any(axis=(1, 3))
    return img_red, trou_red


images = [img]
trous = [trou]
while True:
    img_c, trou_c = reduire(images[-1], trous[-1])
    valides_c = ndimage.binary_erosion(~trou_c, structure=np.ones((taille_patch, taille_patch)), border_value=0)
    if min(img_c.shape[:2]) < taille_min or valides_c.sum() < 200:
        break
    images.append(img_c)
    trous.append(trou_c)
print("pyramide :", [im.shape[:2] for im in images])

guide = None
for n in range(len(images) - 1, -1, -1):
    print("niveau", n, images[n].shape[:2])
    bande_n = None if largeur_bande is None else largeur_bande / 2 ** n
    resultat_n = remplir(images[n], trous[n], guide, bande_n)
    skio.imsave(f"niveau_{n}.png", vers_rgb_uint8(resultat_n))
    if n > 0:
        H, W = trous[n - 1].shape
        guide = resize(resultat_n, (H, W, nb_canaux), order=1, preserve_range=True, anti_aliasing=False)
img = resultat_n

# sources du niveau final, pour la méthode itérative
src = preparer_sources(trou, largeur_bande)
r0, r1, c0, c1 = src[2:6]

#méthode itérative
vote = np.zeros_like(img)
poid = np.zeros_like(img)
s=6
img_trou_ech = [forme[i] for i in range(len(forme)) if (forme[i][0] % s == 0 and forme[i][1] % s == 0)]
print("gogo")
nb_i=5
mediane = False
correspondances = {}   # [PATCHMATCH] (i, j) échantillonné -> centre de sa source, gardé d'une itération à l'autre
for k in range(nb_i):

    skio.imsave(f"resultat_{k}.png", vers_rgb_uint8(img))
    votes_mediane = {}
    vote = np.zeros_like(img)
    poid = np.zeros((height, width))
    liste_votes = []
    img_zone = img[r0:r1, c0:c1]
    for (i, j) in img_trou_ech:
        i_min, i_max = max(0, i - h), min(height, i + h + 1)
        j_min, j_max = max(0, j - h), min(width, j + h + 1)


        pi_min, pi_max = h - (i - i_min), h + (i_max - i)
        pj_min, pj_max = h - (j - j_min), h + (j_max - j)
        mask = np.zeros((taille_patch, taille_patch))
        mask[pi_min:pi_max, pj_min:pj_max] = 1.0
        patch = np.zeros((taille_patch, taille_patch, nb_canaux))
        patch[pi_min:pi_max, pj_min:pj_max] = img[i_min:i_max, j_min:j_max]

        # [PATCHMATCH] candidats de départ : sa propre source à l'itération précédente,
        # et les sources de ses 4 voisins échantillonnés, décalées de l'écart entre eux et lui
        depart = []
        for (vi, vj) in [(i, j), (i - s, j), (i + s, j), (i, j - s), (i, j + s)]:
            if (vi, vj) in correspondances:
                qi_v, qj_v = correspondances[(vi, vj)]
                depart.append((qi_v + i - vi, qj_v + j - vj))

        (qi, qj), d_mse = chercher(img, mask, patch, i, j, src, depart)   # [PATCHMATCH] chercher_source -> chercher
        correspondances[(i, j)] = (qi, qj)                                # [PATCHMATCH]


        di_haut, di_bas = i - i_min, i_max - i
        dj_gauche, dj_droite = j - j_min, j_max - j

        patch_source = img[qi - di_haut : qi + di_bas, qj - dj_gauche : qj + dj_droite]

        if mediane == False :
            liste_votes.append((i_min, i_max, j_min, j_max, patch_source.copy(), d_mse))

        else:
            for d_i in range(patch_source.shape[0]):
                for d_j in range(patch_source.shape[1]):
                    pi, pj = i_min + d_i, j_min + d_j
                    if (pi, pj) not in votes_mediane:
                        votes_mediane[(pi, pj)] = []
                    votes_mediane[(pi, pj)].append(patch_source[d_i, d_j])

    if mediane == False:

        if len(liste_votes) > 0:
            d_tous = np.array([v[5] for v in liste_votes])
            sigma2 = np.percentile(d_tous, 75) + 1e-8
            for (i_min, i_max, j_min, j_max, patch_source, d_v) in liste_votes:
                w = np.exp(-d_v / (2 * sigma2))
                vote[i_min:i_max, j_min:j_max] += patch_source * w
                poid[i_min:i_max, j_min:j_max] += w
            zone_maj = (poid > 0) & trou
            img[zone_maj] = vote[zone_maj] / poid[zone_maj][:, None]
    else:
        for (pi, pj), liste_valeurs in votes_mediane.items():
            if trou[pi, pj]:
                img[pi, pj] = np.median(liste_valeurs, axis=0)
    print("ca roule")

skio.imsave("resultat.png", vers_rgb_uint8(img))
print("résultat enregistré dans resultat.png")