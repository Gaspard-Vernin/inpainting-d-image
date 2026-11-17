import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
import math
from scipy.signal import correlate
from scipy import ndimage

#passe une image lab en rgb 0-255 pour pouvoir l'enregistrer
def vers_rgb_uint8(img_lab):
    rgb = lab2rgb(img_lab)
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)

#fenetre pour dessiner a la souris la zone a effacer, renvoie la liste des pixels du trou (100% by claude)
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

    #dessine (ou efface) un disque autour de la souris
    def paint(event):
        if event.inaxes != ax or not state["drawing"]:
            return
        x, y = int(event.xdata), int(event.ydata)
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= brush_size ** 2
        omega[disk] = not state["erase"]
        overlay.set_data(np.where(omega, 1.0, np.nan))
        fig.canvas.draw_idle()

    #clic gauche = dessiner, clic droit = effacer
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



#image a traiter
chemin="../salon.png"

#on dessine la zone a remplir
forme = get_mask_interactively(chemin, brush_size=40)

#on charge l'image en 3 canaux (on enleve l'alpha, on triple s_i el_le est en gris)
img = skio.imread(chemin).astype(float)
if img.ndim == 2:
    img = img[:, :, None]
img = img[:, :, :3]
if img.shape[2] == 1:
    img = np.repeat(img, 3, axis=2)

shape = img.shape
height, width, nb_canaux = shape

#le trou en ensemble de pixels, pratique pour tester s_i un_ pixel est dedans
forme_origine = set(forme)
forme_set = set(forme)

#parametres
taille_patch = 13   #taille des patchs, impaire pour avoir un centre vraiment au centre
h = (taille_patch - 1) // 2   #demi taille du patch
alpha = 400   #normalisation du terme de données (L entre 0 et 100, sobel de poids 4 donc divise par 400 pour etre inférieur à 1)
mouvements_simples = [(1,0),(0,1),(-1,0),(0,-1)]   #les 4 voisins d'un pixel
largeur_bande = 60   #on cherche les sources a moins de 60 pixels du trou
lambda_dist = 0.02   #poids de la pénalité de distance
poids_guide = 0.3   #poids des pixels comparés au guide (pour la partie multi échelle)
taille_min = 6 * taille_patch   #plus petit côté autorisé pour le niveau le plus réduit

#on enregistre l'image avec le trou en noir pour voir ce qu'on enleve
img_trou = img.copy()
for (i,j) in forme:
    img_trou[i, j, :] = 0
skio.imsave("capture.png", img_trou.astype(np.uint8))

#on passe en cielab, plus proche de la facon dont on voit les couleurs
img = rgb2lab(img / 255.0)

#masque du trou : true la ou il faut remplir
trou = np.zeros((height, width), dtype=bool)
for (i,j) in forme:
    trou[i, j] = True

#un centre est valide s_i to_ut son patch est dans l'image et hors du trou
def sources_valides(trou):
    height, width = trou.shape
    sources = []
    for i in range(h, height - h):
        for j in range(h, width - h):
            ok = True
            for k in range(-h, h + 1):
                for l in range(-h, h + 1):
                    if trou[i + k, j + l]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                sources.append((i, j))
    return sources

#renvoie les patchs sources possibles : entierement hors du trou et dans la bande autour
def trouver_sources(trou, largeur_bande):
    sources = sources_valides(trou)
    #on garde que la bande autour du trou, sauf s'il reste trop peu de sources
    if largeur_bande is not None:
        dist_au_trou = ndimage.distance_transform_edt(~trou)   #distance de chaque pixel au trou
        dans_la_bande = [(i, j) for (i, j) in sources if dist_au_trou[i, j] <= largeur_bande]
        if len(dans_la_bande) >= 200:
            sources = dans_la_bande
    #plus aucune source : le trou est trop grand
    if len(sources) == 0:
        raise ValueError("Aucun patch source valide : le trou est trop grand pour taille_patch=%d" % taille_patch)
    return sources

#calcule la ssd du patch cible avec toutes les sources d'un coup grace a la fft
def chercher_source(img, poids, cible, i_p, j_p, sources):
    height, width, nb_canaux = img.shape
    diag = math.hypot(height, width)         #normalisaion pr que la penalie lambda rese oujours la meme

    #ssd = somme poids*I2 - 2*somme poids*cible*I + somme poids*cible²
    patch_a_changer = cible * poids[:, :, None]
    term3 = np.sum(patch_a_changer * cible)        # = somme des poids * valeur²
    ssd_map = np.zeros((height, width))
    for c in range(nb_canaux):
        im_canal_c = img[:, :, c]
        term1 = correlate(im_canal_c**2, poids, mode='same', method='fft')   #somme poids*I2 pour toutes les positions
        term2 = -2 * correlate(im_canal_c, patch_a_changer[:, :, c], mode='same', method='fft')   #-2*somme poids*cible*I pour toutes les positions
        ssd_map += term1 + term2
    ssd_map += term3   #le terme constant

    #parmi toutes les sources, prend celle qui a le plus petit cout (ssd + pénalité de distance)
    meilleur_cout = float('inf')
    q = None
    mse_q = 0
    for (s_i, s_j) in sources:
        ssd = ssd_map[s_i, s_j]   #ssd de chaque source
        mse = ssd / (poids.sum() * nb_canaux)   #on normalise par le nombre de valeurs comparées
        dist = math.sqrt((s_i - i_p) ** 2 + (s_j - j_p) ** 2) / diag   #distance a la source, normalisée par la diagonale
        cout = round(mse + lambda_dist * 100 ** 2 * dist, 2)
        if cout < meilleur_cout:
            meilleur_cout = cout
            q = (s_i, s_j)
            mse_q = mse
    return q, max(mse_q, 0.0)   #on renvoie aussi la mse, elle sert de poids dans le vote

#gradient de sobel en un pixel (0 sur le bord de l'image)
def sobel(m, i, j):
    if not (1 <= i < m.shape[0] - 1 and 1 <= j < m.shape[1] - 1):
        return 0.0, 0.0

    gradj = (m[i-1, j+1]+2*m[i, j+1]+m[i+1, j+1]) - (m[i-1, j-1]+2*m[i, j-1]+m[i+1, j-1])

    gradi = (m[i+1, j-1]+2*m[i+1, j]+m[i+1, j+1]) - (m[i-1, j-1]+2*m[i-1, j]+m[i-1, j+1])
    return gradi, gradj

#remplit le trou guide = niveau d'avant agrandi (multi échelle), none sinon
def remplir(img, trou, guide=None, largeur_bande=None):
    img = img.copy()   #on travaille sur une copie
    height, width, nb_canaux = img.shape

    #pixels encore a remplir
    forme = [(int(i), int(j)) for (i, j) in np.argwhere(trou)]
    forme_set = set(forme)

    #luminance pour calculer les gradients
    luminance = img[:, :, 0].copy()

    #on remplit la mask de confiance
    confiance = np.ones((height, width))
    for (i,j) in forme:
        confiance[i, j] = 0

    #1 la ou on connait le pixel, 0 sinon (sert pour la normale et les gradients)
    connu = np.ones((height, width))
    for (i,j) in forme:
        connu[i, j] = 0

    sources = trouver_sources(trou, largeur_bande)   #sources possibles a ce niveau
    print("on a", len(sources), "patchs")

    #tant qu'on a encore des pixels pas remplis
    while len(forme_set) > 0:

        #le front = les pixels du trou qui touchent un pixel connu
        frontiere = []
        for (i,j) in forme_set:
            for m in mouvements_simples :
                ni, nj = i + m[0], j + m[1]
                if (ni, nj) not in forme_set and 0 <= ni < height and 0 <= nj < width:
                    frontiere.append((i,j))
                    break
        frontiere.sort()   #on parcourt le front ligne par ligne, pour que les égalités de priorité se départagent toujours pareil

        #calcul de I
        grads = []
        #pour chaque pixel du front on garde le gradient le plus fort de son patch
        for (i,j) in frontiere:
            norme_max = -1
            gradi_max = 0
            gradj_max = 0
            for di in range(-h, h + 1):
                for dj in range(-h, h + 1):
                    a, b = i + di, j + dj
                    #on prend un gradient que s_i se_s 9 voisins sont connus, sinon on lirait l'objet a enlever
                    if 1 <= a < height - 1 and 1 <= b < width - 1 and all((a+x, b+y) not in forme_set for x in (-1, 0, 1) for y in (-1, 0, 1)):
                        gradi, gradj = sobel(luminance, a, b)
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
        #normale au front = gradient du masque des pixels connus, normalisé
        for (i, j) in frontiere:
            gradi_n, gradj_n = sobel(connu, i, j)
            norme = math.sqrt(gradi_n**2 + gradj_n**2)
            if norme == 0:
                n_vals.append((0.0, 0.0))
            else:
                n_vals.append((gradi_n / norme, gradj_n / norme))

        #confiance moyenne sur le patch de chaque pixel du front
        cval = []
        for (i,j) in frontiere:
            somme = 0
            for k in range(-h, h + 1):
                for l in range(-h, h + 1):
                    a, b = i + k, j + l
                    if 0 <= a < height and 0 <= b < width:
                        somme += confiance[a, b]
            cval.append(somme / (taille_patch**2))

        #priorité = confiance * terme de données, on garde le pixel du front qui a la plus grande
        meilleure_priorite = -1
        p_chapeau = None
        C_chapeau = 0
        for i in range(len(frontiere)):
            produit_scalair = abs(np.dot(grads[i], n_vals[i])/alpha)
            prio = cval[i] * (produit_scalair + 0.001)   #+0.001 pour pas que tout soit nul dans les zones plates
            if prio > meilleure_priorite:
                meilleure_priorite = prio
                p_chapeau = frontiere[i]
                C_chapeau = cval[i]
        i_prio, j_prio = p_chapeau

        #le patch autour du pixel choisi, coupé au bord de l'image
        i_min, i_max = max(0, i_prio - h), min(height, i_prio + h + 1)
        j_min, j_max = max(0, j_prio - h), min(width, j_prio + h + 1)

        #les memes bornes mais dans le repere du patch (de 0 a taille_patch)
        pi_min, pi_max = h - (i_prio - i_min), h + (i_max - i_prio)
        pj_min, pj_max = h - (j_prio - j_min), h + (j_max - j_prio)

        #poids de la ssd : 1 pour les pixels connus, 0 pour les autres
        zone_inconnue = np.zeros((taille_patch, taille_patch))
        zone_inconnue[pi_min:pi_max, pj_min:pj_max] = connu[i_min:i_max, j_min:j_max]

        #valeurs du patch cible
        valeurs = img[i_min:i_max, j_min:j_max].copy()
        #multi échelle : les pixels pas encore remplis sont comparés au guide avec un poids plus faible
        if guide is not None:
            inconnus = (connu[i_min:i_max, j_min:j_max] == 0)
            valeurs[inconnus] = guide[i_min:i_max, j_min:j_max][inconnus]
            zone_inconnue[pi_min:pi_max, pj_min:pj_max] += poids_guide * inconnus

        #patch cible complet (des 0 la ou il sort de l'image)
        cible = np.zeros((taille_patch, taille_patch, nb_canaux))
        cible[pi_min:pi_max, pj_min:pj_max] = valeurs

        #q = centre du meilleur patch source
        q, _ = chercher_source(img, zone_inconnue, cible, i_prio, j_prio, sources)

        #on copie la source dans les pixels pas encore remplis et on met tout a jour
        for i in range(-h, h + 1):
            for j in range(-h, h + 1):
                a, b = i_prio + i, j_prio + j
                if 0 <= a < height and 0 <= b < width and (a,b) in forme_set:
                    for c in range(nb_canaux):
                        img[a, b, c] = img[q[0] + i, q[1] + j, c]
                    luminance[a, b] = img[a, b, 0]
                    confiance[a, b] = C_chapeau   #les nouveaux pixels prennent la confiance du patch
                    connu[a, b] = 1
                    forme_set.remove((a,b))

        print("pixels pas encore remplis", len(forme_set))
    return img


#divise l'image et le trou par 2
def reduire(img, trou):
    #flou gaussien en moyennant que les pixels connus pour que l'objet a enlever ne bave pas
    connu = (~trou).astype(float)
    flou_connu = ndimage.gaussian_filter(connu, sigma=1.0)
    img_red = np.zeros_like(img)
    for c in range(img.shape[2]):
        img_red[:, :, c] = ndimage.gaussian_filter(img[:, :, c] * connu, sigma=1.0) / np.maximum(flou_connu, 1e-8)
    img_red = img_red[::2, ::2]   #on garde un pixel sur 2

    #un pixel réduit est dans le trou s_i un_ des 4 pixels de son bloc l'est
    h_shape, w_shape = img_red.shape[:2]
    trou_red = np.zeros((h_shape, w_shape), dtype=bool)
    for i in range(h_shape):
        for j in range(w_shape):
            for k in range(2):
                for l in range(2):
                    if 2*i + k < trou.shape[0] and 2*j + l < trou.shape[1] and trou[2*i + k, 2*j + l]:
                        trou_red[i, j] = True
    return img_red, trou_red


#on réduit tant que l'image reste assez grande et qu'il reste des sources
images = [img]
trous = [trou]
while True:
    img_c, trou_c = reduire(images[-1], trous[-1])
    if min(img_c.shape[:2]) < taille_min or len(sources_valides(trou_c)) < 200:
        break
    images.append(img_c)
    trous.append(trou_c)
print("pyramide :", [im.shape[:2] for im in images])

#on remplit du plus petit niveau au plus grand, chaque niveau est guidé par le précédent agrandi
guide = None
h_niveau0 = h
for n in range(len(images) - 1, -1, -1):
    print("niveau", n, images[n].shape[:2])
    bande_n = largeur_bande / (n+1) #on redui la aille de la bande.
    h = max(3, h_niveau0 //(2**n))    
    taille_patch = 2 * h + 1 
    resultat_n = remplir(images[n], trous[n], guide, bande_n)
    skio.imsave(f"niveau_{n}.png", vers_rgb_uint8(resultat_n))   #on enregistre chaque niveau pour voir
    if n > 0:
        h_shape, w_shape = trous[n - 1].shape
        guide = resize(resultat_n, (h_shape, w_shape, nb_canaux), order=1, preserve_range=True, anti_aliasing=False)   #le résultat agrandi sert de guide au niveau suivant
img = resultat_n

#sources du niveau final, pour la méthode itérative
sources = trouver_sources(trou, largeur_bande)

#méthode itérative
s=6   #on prend un pixel sur s dans chaque direction
#pixels du trou échantillonnés
img_trou_ech = [forme[i] for i in range(len(forme)) if (forme[i][0] % s == 0 and forme[i][1] % s == 0)]
print("gogo")
nb_i=5   #nombre d'itérations
mediane = False   #true pour faire la médiane des votes au lieu de la moyenne
for k in range(nb_i):
    skio.imsave(f"resultat_{k}.png", vers_rgb_uint8(img))   #on enregistre chaque itération
    #accumulateurs du vote 
    votes_mediane = {}
    vote = np.zeros_like(img)
    poid = np.zeros((height, width))
    liste_votes = []
    #pour chaque pixel échantillonné on cherche la source de tout son patch (il est rempli maintenant)
    for (i, j) in img_trou_ech:
        i_min, i_max = max(0, i - h), min(height, i + h + 1)
        j_min, j_max = max(0, j - h), min(width, j + h + 1)

        pi_min, pi_max = h - (i - i_min), h + (i_max - i)
        pj_min, pj_max = h - (j - j_min), h + (j_max - j)
        #ici tous les pixels du patch comptent
        mask = np.zeros((taille_patch, taille_patch))
        mask[pi_min:pi_max, pj_min:pj_max] = 1.0
        patch = np.zeros((taille_patch, taille_patch, nb_canaux))
        patch[pi_min:pi_max, pj_min:pj_max] = img[i_min:i_max, j_min:j_max]

        (qi, qj), d_mse = chercher_source(img, mask, patch, i, j, sources)   #meilleure source du patch entier

        di_haut, di_bas = i - i_min, i_max - i
        dj_gauche, dj_droite = j - j_min, j_max - j
        #le patch source, coupé pareil que le patch cible
        patch_source = img[qi - di_haut : qi + di_bas, qj - dj_gauche : qj + dj_droite]

        #moyenne : on garde le patch source et son erreur pour le vote
        if mediane == False :
            liste_votes.append((i_min, i_max, j_min, j_max, patch_source.copy(), d_mse))

        else:
            #médiane : on garde toutes les valeurs proposées pour chaque pixel
            for d_i in range(patch_source.shape[0]):
                for d_j in range(patch_source.shape[1]):
                    pi, pj = i_min + d_i, j_min + d_j
                    if (pi, pj) not in votes_mediane:
                        votes_mediane[(pi, pj)] = []
                    votes_mediane[(pi, pj)].append(patch_source[d_i, d_j])

    #chaque pixel du trou = moyenne pondérée des valeurs proposées
    if mediane == False:
        if len(liste_votes) > 0:
            d_tous = np.array([v[5] for v in liste_votes])
            sigma2 = np.percentile(d_tous, 25) + 1e-8    
            for (i_min, i_max, j_min, j_max, patch_source, d_v) in liste_votes:
                w = np.exp(-d_v / (2 * sigma2))
                vote[i_min:i_max, j_min:j_max] += patch_source * w
                poid[i_min:i_max, j_min:j_max] += w
            zone_maj = (poid > 0) & trou   #on ne touche qu'aux pixels du trou
            img[zone_maj] = vote[zone_maj] / poid[zone_maj][:, None]
    else:
        #chaque pixel du trou = médiane des valeurs proposées
        for (pi, pj), liste_valeurs in votes_mediane.items():
            if trou[pi, pj]:
                img[pi, pj] = np.median(liste_valeurs, axis=0)
    print("ca roule")

skio.imsave("resultat.png", vers_rgb_uint8(img))
print("résultat enregistré dans resultat.png")