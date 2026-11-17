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
forme = get_mask_interactively(chemin, brush_size=30)


#on charge l'image en 3 canaux (on enleve l'alpha, on triple si elle est en gris)
img = skio.imread(chemin).astype(float)
if img.ndim == 2:
    img = img[:, :, None]
img = img[:, :, :3]

if img.shape[2] == 1:
    img = np.repeat(img, 3, axis=2)

shape = img.shape
height, width, nb_canaux = shape

#le trou en ensemble de pixels, pratique pour tester si un pixel est dedans
forme_origine = set(forme)
forme_set = set(forme)

#parametres
taille_patch = 13   #taille des patchs, impaire pour avoir un centre vraiment au centre
h = (taille_patch - 1) // 2   #demi taille du patch

alpha = 400   #normalisation du terme de données (L entre 0 et 100, sobel de poids 4 donc divise par 400 pour etre inférieur à 1)

largeur_bande = 120   #on cherche les sources a moins de largeur_bande pixels du trou
lambda_dist = 0.006   #poids de la pénalité de distance

poids_guide = 0.1   #poids des pixels comparés au guide (pour la partie multi échelle)
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

#renvoie les patchs sources possibles : entierement hors du trou et dans la bande autour
def preparer_sources(trou, largeur_bande):
    height, width = trou.shape
    #un centre est valide si tout son patch est hors du trou (érosion par un carré de la taille d'un patch)
    valides = ndimage.binary_erosion(~trou, structure=np.ones((taille_patch, taille_patch)), border_value=0)

    masque_sources = valides
    #on garde que la bande autour du trou, sauf s'il reste trop peu de sources
    if largeur_bande is not None:
        dist_au_trou = ndimage.distance_transform_edt(~trou)   #distance de chaque pixel au trou
        dans_la_bande = dist_au_trou <= largeur_bande
        if (valides & dans_la_bande).sum() >= 200:
            masque_sources = valides & dans_la_bande

    #plus aucune source : le trou est trop grand
    if masque_sources.sum() == 0:
        raise ValueError("Aucun patch source valide : le trou est trop grand pour taille_patch=%d" % taille_patch)

    #rectangle qui contient toutes les sources, on calculera la ssd que dedans
    a, b = np.nonzero(masque_sources)
    r0, r1 = max(0, a.min() - h), min(height, a.max() + h + 1)
    c0, c1 = max(0, b.min() - h), min(width, b.max() + h + 1)

    diag = math.hypot(height, width)         #normalisaion pr que la penalie lambda rese oujours la meme

    #coordonnées des centres valides
    valides = np.argwhere(masque_sources)
    valides_i, valides_j = valides[:, 0], valides[:, 1]
    return valides_i, valides_j, r0, r1, c0, c1, diag


#parmi toutes les sources, prend celle qui a le plus petit cout (ssd + pénalité de distance)
def meilleure_source(ssd_map_c, nb_pixels_connus, i_p, j_p, src):
    valides_i, valides_j, r0, r1, c0, c1, diag = src
    nb_canaux = 3
    ssd = ssd_map_c[valides_i - r0, valides_j - c0 ]   #ssd de chaque source
    mse = ssd / (nb_pixels_connus * nb_canaux)   #on normalise par le nombre de valeurs comparées
    dist = np.sqrt((valides_i - i_p) ** 2 + (valides_j - j_p) ** 2) / diag   #distance a la source, normalisée par la diagonale
    cout = np.round(mse + lambda_dist * 100 ** 2 * dist, 2)
    k = np.argmin(cout)
    return (int(valides_i[k]), int(valides_j[k])), max(float(mse[k]), 0.0)   #on renvoie aussi la mse, elle sert de poids dans le vote



#calcule la ssd du patch cible avec toutes les sources d'un coup grace a la fft
def chercher_source(img, poids, cible, i_p, j_p, src, depart=None):   #depart sert que pour patchmatch, ici on l'ignore
    r0, r1, c0, c1 = src[2:6]
    nb_canaux = img.shape[2]

    #ssd = somme poids*I² - 2*somme poids*cible*I + somme poids*cible²
    patch_a_changer = cible * poids[:, :, None]
    term3 = np.sum(patch_a_changer * cible)        # = somme des poids * valeur²

    img_zone = img[r0:r1, c0:c1]
    ssd_map = np.zeros((r1 - r0, c1 - c0))
    for c in range(nb_canaux):
        im_canal_c = img_zone[:, :, c]
        term1 = correlate(im_canal_c**2, poids, mode='same', method='fft')   #somme poids*I2 pour toutes les positions
        term2 = -2 * correlate(im_canal_c, patch_a_changer[:, :, c], mode='same', method='fft')   #-2*somme poids*cible*I pour toutes les positions
        ssd_map += term1 + term2
    ssd_map += term3   #le terme constant

    return meilleure_source(ssd_map, poids.sum(), i_p, j_p, src)


nb_aleatoires = 30     #candidats tirés au hasard au départ
nb_essais = 16         #candidats testés a chaque rayon de la recherche aléatoire
nb_tours = 2           #nombre de passes de recherche aléatoire

#pour chaque candidat (ci, cj) : est ce que c'est bien le centre d'une source valide
#est_source c'est une image de booléens, true la ou il y a une source
def est_valide(ci, cj, est_source):
    height, width = est_source.shape
    dedans = (ci >= 0) & (ci < height) & (cj >= 0) & (cj < width)   #d'abord il faut etre dans l'image
    ok = np.zeros(len(ci), dtype=bool)
    ok[dedans] = est_source[ci[dedans], cj[dedans]]   #puis on regarde dans le masque des sources
    return ok

#cout de chaque patch candidat (comme dans meilleure_source)
def couts(img, ci, cj, poids, cible, i_p, j_p, diag):
    nb_canaux = img.shape[2]
    cout = np.zeros(len(ci))
    mse = np.zeros(len(ci))
    for k in range(len(ci)):
        patch = img[ci[k] - h : ci[k] + h + 1, cj[k] - h : cj[k] + h + 1]   #le patch source candidat
        ssd = np.sum(poids[:, :, None] * (patch - cible) ** 2)
        mse[k] = ssd / (poids.sum() * nb_canaux)
        dist = math.sqrt((ci[k] - i_p) ** 2 + (cj[k] - j_p) ** 2) / diag
        cout[k] = mse[k] + lambda_dist * 100 ** 2 * dist
    return cout, mse

#au lieu de tester toutes les sources on en teste juste quelques unes bien choisies (patchmatch)
#le détail du fonctionnement de cette fonction est décrit dans le rapport
def chercher_source_patchmatch(img, poids, cible, i_p, j_p, src, depart=None):
    valides_i, valides_j, r0, r1, c0, c1, diag = src
    height, width, nb_canaux = img.shape
    #masque des sources on me true au centre de chaque source valide
    est_source = np.zeros((height, width), dtype=bool)
    est_source[valides_i, valides_j] = True
    #on part de sources au hasard + celles de depart
    k = np.random.randint(0, len(valides_i), nb_aleatoires)
    ci, cj = valides_i[k], valides_j[k]
    if depart is not None and len(depart) > 0:
        depart = np.unique(np.asarray(depart, dtype=int).reshape(-1, 2), axis=0)
        ok = est_valide(depart[:, 0], depart[:, 1], est_source)
        ci = np.concatenate([ci, depart[ok, 0]])
        cj = np.concatenate([cj, depart[ok, 1]])
    c, m = couts(img, ci, cj, poids, cible, i_p, j_p, diag)
    b = np.argmin(c)
    bi, bj, bc, bm = ci[b], cj[b], c[b], m[b]

    #on cherche autour du meilleur, le rayon est divisé par 2 a chaque fois
    for tour in range(nb_tours):
        r = max(height, width)
        while r >= 1:
            #on garde seulement les sources valides qui sont dans le carré de rayon r autour du meilleur
            dans_carre = (np.abs(valides_i - bi) <= r) & (np.abs(valides_j - bj) <= r)
            proches_i = valides_i[dans_carre]
            proches_j = valides_j[dans_carre]
            if len(proches_i) > 0:
                #on en tire nb_essais au hasard parmi elles, comme ca aucun essai n'est perdu hors des sources
                k = np.random.randint(0, len(proches_i), nb_essais)
                ci, cj = proches_i[k], proches_j[k]
                c, m = couts(img, ci, cj, poids, cible, i_p, j_p, diag)
                b = np.argmin(c)
                if c[b] < bc:
                    bi, bj, bc, bm = ci[b], cj[b], c[b], m[b]
            r = r // 2

    return (int(bi), int(bj)), max(float(bm), 0.0)


#ici, on peut choisir si on décide de chercher le meilleur patch avec patchmatch ou avec la methode par fourrier
chercher = chercher_source_patchmatch      #patchmatch, approché mais plus rapide


#remplit le trou guide = niveau d'avant agrandi, none sinon
def remplir(img, trou, guide=None, largeur_bande=None, origine_grossier=None):   #origine_grossier = sources trouvées au niveau d'avant (pour patchmatch)
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

    reste = trou.copy()   #le trou qui reste a remplir

    #pour chaque pixel rempli, le pixel dont on l'a copié (ca sert de départ a patchmatch)
    origine = np.full((height, width, 2), -1)

    src = preparer_sources(trou, largeur_bande)   #sources possibles a ce niveau

    #tant qu'on a encore des pixels pas remplis
    while len(forme_set) > 0:

        #le front = les pixels du trou qui touchent un pixel connu (le trou moins son érodé)
        croix = ndimage.generate_binary_structure(2, 1)
        masque_front = reste & ~ndimage.binary_erosion(reste, structure=croix, border_value=1)
        frontiere = [(int(i), int(j)) for (i, j) in np.argwhere(masque_front)]
        fi, fj = np.nonzero(masque_front)    #le front en tableaux, dans le meme ordre que frontiere
        #calcul de I
        #sobel de toute l'image d'un coup
        gi = ndimage.sobel(luminance, axis=0)
        gj = ndimage.sobel(luminance, axis=1)
        #on prend un gradient que si ses 9 voisins sont connus, sinon on lirait l'objet a enlever (érosion par un carré 3x3)
        grad_ok = ndimage.binary_erosion(connu.astype(bool), structure=np.ones((3, 3)), border_value=0)
        norme_img = np.where(grad_ok, np.sqrt(gi**2 + gj**2), -1.0)      #-1 si on peut pas l'utiliser
        #pour chaque pixel du front on regarde les normes de tout son patch (-2 en dehors de l'image)
        offs = np.arange(-h, h + 1)
        pad = np.pad(norme_img, h, constant_values=-2.0)
        fenetres = pad[(fi + h)[:, None, None] + offs[None, :, None], (fj + h)[:, None, None] + offs[None, None, :]]
        #on garde le gradient le plus fort du patch
        k_max = fenetres.reshape(len(fi), -1).argmax(axis=1)
        a_max = fi + offs[k_max // taille_patch]
        b_max = fj + offs[k_max % taille_patch]
        valide = norme_img[a_max, b_max] >= 0   #si aucun pixel utilisable le gradient reste a (0, 0)
        gradi_max = np.where(valide, gi[a_max, b_max], 0.0)
        gradj_max = np.where(valide, gj[a_max, b_max], 0.0)
        #on le tourne de 90 degré ce qui revient a faire
        tang_i, tang_j = -gradj_max, gradi_max

        #calcul de n
        #normale au front = gradient du masque des pixels connus, normalisé
        ni = ndimage.sobel(connu, axis=0)[fi, fj]
        nj = ndimage.sobel(connu, axis=1)[fi, fj]
        bord = (fi < 1) | (fi >= height - 1) | (fj < 1) | (fj >= width - 1)   #au bord de l'image la normale reste a (0, 0)
        ni = np.where(bord, 0.0, ni)
        nj = np.where(bord, 0.0, nj)
        norme_n = np.sqrt(ni**2 + nj**2)
        non_nul = norme_n > 0
        ni = np.where(non_nul, ni / np.where(non_nul, norme_n, 1.0), 0.0)
        nj = np.where(non_nul, nj / np.where(non_nul, norme_n, 1.0), 0.0)

        #confiance moyenne sur le patch de chaque pixel du front
        carte_conf = ndimage.uniform_filter(confiance, size=taille_patch, mode='constant', cval=0.0)
        cval = carte_conf[fi, fj]

        #priorité = confiance * terme de données, on garde le pixel du front qui a la plus grande
        produit_scalair = np.abs((tang_i * ni + tang_j * nj) / alpha)
        prio = cval * (produit_scalair + 0.001)   #+0.001 pour pas que tout soit nul dans les zones plates
        k_prio = np.argmax(prio)
        p_chapeau = frontiere[k_prio]
        C_chapeau = cval[k_prio]
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

        #pour patchmatch les candidats de départ c'est les sources des pixels deja remplis du patch,
        #décalées de la position du pixel dans le patch
        zone_origine = origine[i_min:i_max, j_min:j_max].reshape(-1, 2)
        decal_i, decal_j = np.mgrid[i_min - i_prio:i_max - i_prio, j_min - j_prio:j_max - j_prio]
        rempli = zone_origine[:, 0] >= 0
        depart = zone_origine[rempli] - np.stack([decal_i.ravel(), decal_j.ravel()], axis=1)[rempli]

        #multi échelle : on ajoute aussi la source trouvée au niveau d'avant pour ce pixel
        #(le pixel (i, j) ici correspond au pixel (i//2, j//2) du niveau d'avant, on remultiplie par 2)
        if origine_grossier is not None:
            oi, oj = origine_grossier[i_prio // 2, j_prio // 2]
            if oi >= 0:
                depart = list(depart)
                depart.append((2 * oi + i_prio % 2, 2 * oj + j_prio % 2))

        #q = centre du meilleur patch source
        q, _ = chercher(img, zone_inconnue, cible, i_prio, j_prio, src, depart)

        #on copie la source dans les pixels pas encore remplis et on met tout a jour
        for i in range(-h, h + 1):
            for j in range(-h, h + 1):
                a, b = i_prio + i, j_prio + j
                if 0 <= a < height and 0 <= b < width and (a,b) in forme_set:
                    for c in range(nb_canaux):
                        img[a, b, c] = img[q[0] + i, q[1] + j, c]
                    origine[a, b] = (q[0] + i, q[1] + j)   #on retient d'où ce pixel a été copié

                    luminance[a, b] = img[a, b, 0]
                    confiance[a, b] = C_chapeau   #les nouveaux pixels prennent la confiance du patch
                    connu[a, b] = 1
                    reste[a, b] = False
                    forme_set.remove((a,b))

        print("pixels pas encore remplis", len(forme_set))
    return img, origine   #on renvoie aussi d'où vient chaque pixel, ca sert de départ a patchmatch ensuite



#divise l'image et le trou par 2
def reduire(img, trou):
    #flou gaussien en moyennant que les pixels connus, pour que l'objet a enlever bave pas
    connu = (~trou).astype(float)
    flou_connu = ndimage.gaussian_filter(connu, sigma=1.0)
    img_red = np.zeros_like(img)
    for c in range(img.shape[2]):
        img_red[:, :, c] = ndimage.gaussian_filter(img[:, :, c] * connu, sigma=1.0) / np.maximum(flou_connu, 1e-8)
    img_red = img_red[::2, ::2]   #on garde un pixel sur 2

    #un pixel réduit est dans le trou si un des 4 pixels de son bloc l'est
    h_shape, w_shape = img_red.shape[:2]
    trou_pad = np.zeros((2 * h_shape, 2 * w_shape), dtype=bool)
    trou_pad[:trou.shape[0], :trou.shape[1]] = trou
    trou_red = trou_pad.reshape(h_shape, 2, w_shape, 2).any(axis=(1, 3))
    return img_red, trou_red


#la pyramide : on réduit tant que l'image reste assez grande et qu'il reste des sources
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

#on remplit du plus petit niveau au plus grand, chaque niveau est guidé par le précédent agrandi
chercher = chercher_source   #le remplissage se fait avec la recherche exhaustive par fourier (patchmatch y marche mal)
guide = None
origine_grossier = None   #sources du niveau d'avant, none au premier niveau
h_niveau0 = h
for n in range(len(images) - 1, -1, -1):
    print("niveau", n, images[n].shape[:2])
    bande_n = largeur_bande / (n+1) #on redui la aille de la bande.
    #on redui la aille des pachs car l'image es plus peie
    h = max(3, h_niveau0 // 2 ** n)   
    taille_patch = 2 * h + 1
    resultat_n, origine_n = remplir(images[n], trous[n], guide, bande_n, origine_grossier)
    origine_grossier = origine_n   #elles serviront de départ a patchmatch au niveau suivant
    skio.imsave(f"niveau_{n}.png", vers_rgb_uint8(resultat_n))   #on enregistre chaque niveau pour voir
    if n > 0:
        H, W = trous[n - 1].shape
        guide = resize(resultat_n, (H, W, nb_canaux), order=1, preserve_range=True, anti_aliasing=False)   #le résultat agrandi sert de guide au niveau suivant
img = resultat_n

chercher = chercher_source_patchmatch   #la phase itérative se fait avec patchmatch
#sources du niveau final, pour la méthode itérative
src = preparer_sources(trou, largeur_bande)
r0, r1, c0, c1 = src[2:6]

#méthode itérative
#accumulateurs du vote
vote = np.zeros_like(img)
poid = np.zeros_like(img)
s=6   #on prend un pixel sur s dans chaque direction
#pixels du trou échantillonnés
img_trou_ech = [forme[i] for i in range(len(forme)) if (forme[i][0] % s == 0 and forme[i][1] % s == 0)]
print("gogo")
nb_i=5   #nombre d'itérations
mediane = False  #true pour faire la médiane des votes au lieu de la moyenne
#la source de chaque patch, gardée d'une itération a l'autre (pour patchmatch)
correspondances = {}
#au départ on reprend les sources trouvées par remplir : le pixel (i, j) a été copié depuis origine_n[i, j],
#donc le patch centré en origine_n[i, j] est un bon candidat pour le patch centré en (i, j)
for (i, j) in img_trou_ech:
    if origine_n[i, j, 0] >= 0:
        correspondances[(i, j)] = (int(origine_n[i, j, 0]), int(origine_n[i, j, 1]))
for k in range(nb_i):

    skio.imsave(f"resultat_{k}.png", vers_rgb_uint8(img))   #on enregistre chaque itération
    votes_mediane = {}
    vote = np.zeros_like(img)
    poid = np.zeros((height, width))
    liste_votes = []
    img_zone = img[r0:r1, c0:c1]
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

        #pour patchmatch les candidats de départ c'est sa propre source à l'itération précédente,
        #et les sources de ses 4 voisins (avec une strid définie)
        depart = []
        for (vi, vj) in [(i, j), (i - s, j), (i + s, j), (i, j - s), (i, j + s)]:
            if (vi, vj) in correspondances:
                qi_v, qj_v = correspondances[(vi, vj)]
                depart.append((qi_v + i - vi, qj_v + j - vj))

        (qi, qj), d_mse = chercher(img, mask, patch, i, j, src, depart)   #meilleure source du patch entier
        correspondances[(i, j)] = (qi, qj)

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

    #chaque pixel du trou = moyenne des valeurs proposées, pondérée par exp(-erreur/2sigma²)
    if mediane == False:

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
