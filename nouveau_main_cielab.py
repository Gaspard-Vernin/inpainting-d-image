# =====================================================================
#  new_main_lab.py
#  Inpainting par exemplaires (Criminisi et al. 2003) + raffinement type Wexler
#  Version : tout le calcul se fait dans l'espace couleur CIE Lab (comme dans l'article)
#
#  LÉGENDE DES MODIFICATIONS (cherche ces tags dans le code, Ctrl+F) :
#
#   [LAB]              NOUVEAU : conversion en espace CIE Lab (chaque endroit modifié est expliqué)
#   [MASQUE BINAIRE]   la SSD utilise un masque 0/1 (explication détaillée à l'endroit du calcul)
#   [CORRIGÉ]          correction de bug (versions précédentes)
#   [OPT 1..5]         optimisations de base (résultat identique, plus rapide)
#   [AJOUT: BANDE]     sources restreintes à une bande autour du trou (+ FFT sur la zone utile seulement)
#   [AJOUT: DISTANCE]  pénalité de distance spatiale dans le coût de patch
#   [AJOUT: WEXLER]    vote pondéré à la Wexler dans la phase itérative (activé par défaut)
#
#  Les anciens morceaux de code remplacés sont gardés en commentaire "# ANCIEN :".
#  Tout ce qui n'est pas tagué est EXACTEMENT ton code d'origine.
# =====================================================================

import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage.color import rgb2lab, lab2rgb   # [LAB] import ajouté : conversions RGB <-> CIE Lab
import math
from scipy.signal import correlate
from scipy import ndimage   # [OPT 3][OPT 5][AJOUT: BANDE] import ajouté

# [LAB] Fonction ajoutée : convertit une image Lab en image RGB uint8 pour l'enregistrer.
#   Avant, on faisait  np.clip(img, 0, 255).astype(np.uint8)  car img était en RGB.
#   Maintenant img est en Lab (L entre 0 et 100, a et b entre -128 et 127 environ) : il faut repasser en RGB
#   avant d'écrire un .png. lab2rgb renvoie des valeurs entre 0 et 1 (et ramène dans la gamme sRGB les couleurs
#   qui en sortent, ce qui peut arriver après une moyenne ou une médiane).
def vers_rgb_uint8(img_lab):
    rgb = lab2rgb(img_lab)
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)

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



chemin="onio (1).jpg"

forme = get_mask_interactively(chemin, brush_size=40)


img = skio.imread(chemin).astype(float)
if img.ndim == 2:
    img = img[:, :, None]
img = img[:, :, :3]
# [LAB] rgb2lab a besoin de 3 canaux : une image en niveaux de gris est dupliquée sur 3 canaux
#   (avant, ton code acceptait 1 canal ; on garde cette compatibilité).
if img.shape[2] == 1:
    img = np.repeat(img, 3, axis=2)

shape = img.shape
height, width, nb_canaux = shape

forme_origine = set(forme)
forme_set = set(forme)
# [OPT 1] supprimé : print(forme_origine)   <- affichait des milliers de tuples et ralentissait le script
taille_patch = 13
h = (taille_patch - 1) // 2
# [LAB] ANCIEN : alpha = 4 * 255
#   alpha normalise le gradient dans le terme de données D(p) = |gradient . normale| / alpha pour que D reste dans [0,1].
#   Le gradient est maintenant calculé sur la luminance L, qui va de 0 à 100 (et non plus de 0 à 255),
#   et le Sobel multiplie par 4 -> gradient max = 4 * 100.
alpha = 4 * 100
mouvements_simples = [(1,0),(0,1),(-1,0),(0,-1)]   # [OPT 5] n'est plus utilisé (la frontière est vectorisée), gardé par commodité

# ---------------------------------------------------------------------
# [AJOUT: BANDE] [AJOUT: DISTANCE] [LAB] PARAMÈTRES (à régler ici)
# ---------------------------------------------------------------------
largeur_bande = 60   # [AJOUT: BANDE] on ne cherche les patchs sources que dans les pixels situés à
                     #   moins de 'largeur_bande' pixels du trou. Mets None pour chercher dans toute l'image.
                     #   (doit être > h+1 sinon il n'y a aucune source ; si trop peu de sources, repli automatique sur toute l'image)
lambda_dist = 0.02   # [AJOUT: DISTANCE] force de la pénalité de distance. 0 = désactivée.
                     #   Coût = MSE + lambda_dist * ECHELLE_COULEUR^2 * (distance / diagonale de l'image)
                     #   [LAB] avec ECHELLE_COULEUR = 100 : un patch situé à 10% de la diagonale coûte +20 de MSE
                     #   (soit un écart moyen d'environ 4,5 unités Lab par canal). ANCIEN en RGB : 255^2 (+130).
ECHELLE_COULEUR = 100.0   # [LAB] ANCIEN : 255. Amplitude d'un canal : la luminance L va de 0 à 100 (en RGB c'était 0..255).
# ---------------------------------------------------------------------

img_trou = img.copy()
for (i,j) in forme:
    img_trou[i, j, :] = 0
skio.imsave("capture.png", img_trou.astype(np.uint8))

# [LAB] >>>>>>> CONVERSION RGB -> CIE Lab (c'est ici que tout bascule en Lab) <<<<<<<
#   À partir de cette ligne, 'img' contient (L, a, b) et non plus (R, G, B) :
#     L = luminosité (0 à 100), a = axe vert<->rouge, b = axe bleu<->jaune (environ -128 à +127).
#   Pourquoi : en RGB, deux couleurs à distance euclidienne égale peuvent paraître très différentes ou presque
#   identiques à l'œil. En Lab, la distance euclidienne correspond à peu près à la différence PERÇUE
#   (c'est la propriété d'"uniformité perceptuelle" que cite l'article). Donc la SSD compare mieux les patchs.
#   Tout ce qui suit (SSD, copie des pixels, vote de Wexler, médiane) travaille donc en Lab ; seul l'enregistrement
#   des images repasse en RGB (fonction vers_rgb_uint8). La capture "capture.png" ci-dessus reste en RGB, avant conversion.
#   rgb2lab attend du RGB dans [0,1], d'où la division par 255.
img = rgb2lab(img / 255.0)

#on met l'image en niveaux de gris (plus tard on pourra le faire sur des couleurs mais c plus simple pr mtn)
# [LAB] gris = canal L (luminance). ANCIEN : gris = img.mean(axis=2)  (moyenne de R, G, B).
#   En Lab, moyenner L, a et b n'a aucun sens (ce ne sont pas trois "intensités"). Pour les isophotes (gradient
#   d'intensité) on prend donc la luminance L, qui joue le rôle du niveau de gris. .copy() pour avoir un tableau
#   indépendant (on le met à jour pixel par pixel pendant le remplissage).
gris = img[:, :, 0].copy()
# ANCIEN (avant [OPT 2]) :
# gris = np.zeros((height, width))
# for i in range(height):
#     for j in range(width):
#         gris[i, j]=0
#         for c in range(nb_canaux):
#             gris[i, j] += img[i, j, c]
#         gris[i, j] /= nb_canaux

#on remplit la mask de confiance
confiance = np.ones((height, width))
for (i,j) in forme:
    confiance[i, j] = 0

# [CORRIGÉ][MASQUE BINAIRE] masque binaire des pixels connus (0/1) : sert au SSD, pendant que "confiance" ne sert qu'à la priorité
connu = np.ones((height, width))
for (i,j) in forme:
    connu[i, j] = 0

# [CORRIGÉ] masque booléen du trou d'origine : sert à ne modifier QUE le trou dans la phase itérative
trou = np.zeros((height, width), dtype=bool)
for (i,j) in forme:
    trou[i, j] = True

# [OPT 5] masque booléen des pixels PAS ENCORE remplis (équivalent de forme_set, mais en tableau numpy)
reste = trou.copy()

#on presave tout les patchs
print("ok")
# [OPT 3] un pixel (i,j) est une source valide si sa fenêtre 13x13 ne touche aucun pixel du trou
#         ET est entièrement dans l'image (border_value=0 : l'extérieur de l'image compte comme "non valide").
#         Même résultat que les 4 boucles imbriquées d'origine, en une fraction de seconde.
valides = ndimage.binary_erosion(~trou, structure=np.ones((taille_patch, taille_patch)), border_value=0)
# ANCIEN :
# sources = []
# for i in range(h,height-h):
#     for j in range(h,width-h):
#         ok = True
#         for k in range(-h,h+1):
#             for l in range(-h,h+1):
#                 if (i+k,j+l) in forme_origine:
#                     ok = False
#                     break
#             if not ok:
#                 break
#         if ok:
#             sources.append((i,j))

# [AJOUT: BANDE] on ne garde que les sources proches du trou (distance euclidienne au trou <= largeur_bande)
masque_sources = valides
if largeur_bande is not None:
    dist_au_trou = ndimage.distance_transform_edt(~trou)
    dans_la_bande = dist_au_trou <= largeur_bande
    if (valides & dans_la_bande).sum() >= 200:
        masque_sources = valides & dans_la_bande
    else:
        print("[BANDE] bande trop étroite (< 200 sources) -> on cherche dans toute l'image")
if masque_sources.sum() == 0:
    raise ValueError("Aucun patch source valide : le trou est trop grand pour taille_patch=%d" % taille_patch)

sources = [(int(i), int(j)) for (i, j) in np.argwhere(masque_sources)]   # [OPT 3] gardé seulement pour l'affichage du nombre
print("on a",len(sources),"patchs")

# [AJOUT: BANDE] zone de l'image utile pour la recherche : boîte englobante des sources + marge h.
#   Les corrélations FFT ne sont calculées que sur cette zone (au lieu de toute l'image) -> beaucoup plus rapide
#   quand la bande est petite devant l'image. Les résultats aux positions sources sont identiques.
_ys, _xs = np.nonzero(masque_sources)
r0, r1 = max(0, _ys.min() - h), min(height, _ys.max() + h + 1)
c0, c1 = max(0, _xs.min() - h), min(width, _xs.max() + h + 1)
masque_src_c = masque_sources[r0:r1, c0:c1]      # masque des sources, dans le repère de la zone
YY, XX = np.mgrid[r0:r1, c0:c1]                   # coordonnées (dans l'image) de chaque pixel de la zone
diag = math.hypot(height, width)                  # [AJOUT: DISTANCE] diagonale de l'image (normalisation)

def meilleure_source(ssd_map_c, nb_pixels_connus, i_p, j_p):
    """
    [OPT 4] [AJOUT: BANDE] [AJOUT: DISTANCE] [LAB]
    Choisit le meilleur patch source pour le patch centré en (i_p, j_p).
      ssd_map_c        : carte des SSD calculée sur la zone [r0:r1, c0:c1] (SSD en unités Lab^2)
      nb_pixels_connus : nombre de pixels comparés dans le patch (pour normaliser en MSE)
    Retourne ((qi, qj), mse) : centre du meilleur patch (repère image) et son MSE (sans pénalité).
    """
    # SSD -> erreur quadratique moyenne par valeur (comparable entre patchs de tailles différentes)
    mse = ssd_map_c / (nb_pixels_connus * nb_canaux)
    # [AJOUT: DISTANCE] pénalité proportionnelle à la distance entre le patch cible et le candidat
    dist = np.sqrt((YY - i_p) ** 2 + (XX - j_p) ** 2) / diag
    # [LAB] ANCIEN : cout = mse + lambda_dist * 255 ** 2 * dist
    #   Le MSE est maintenant en unités Lab^2 : la pénalité utilise donc l'échelle Lab (100^2) et non 255^2,
    #   sinon la pénalité serait ~6 fois trop forte par rapport au MSE.
    cout = mse + lambda_dist * ECHELLE_COULEUR ** 2 * dist
    # [OPT 4] argmin numpy ; [AJOUT: BANDE] seules les sources valides de la bande sont candidates
    cout = np.where(masque_src_c, np.round(cout, 2), np.inf)
    ci, cj = np.unravel_index(np.argmin(cout), cout.shape)
    return (int(ci + r0), int(cj + c0)), max(float(mse[ci, cj]), 0.0)

def sobel(m, i, j):
    if not (1 <= i < m.shape[0] - 1 and 1 <= j < m.shape[1] - 1):
        return 0.0, 0.0

    gradj = (m[i-1, j+1]+2*m[i, j+1]+m[i+1, j+1]) - (m[i-1, j-1]+2*m[i, j-1]+m[i+1, j-1])

    gradi = (m[i+1, j-1]+2*m[i+1, j]+m[i+1, j+1]) - (m[i-1, j-1]+2*m[i-1, j]+m[i-1, j+1])
    return gradi, gradj
#tant qu'on a encore des pixels pas remplis
while len(forme_set) > 0:

    #calcul de la frontiere
    # [OPT 5] frontière = pixels pas remplis qui ont au moins un voisin (4-voisinage, dans l'image) déjà rempli.
    #         Érosion du masque "reste" avec une croix ; l'extérieur de l'image compte comme "reste" (border_value=1)
    #         pour reproduire ta condition 0 <= ni < height.
    croix = ndimage.generate_binary_structure(2, 1)
    masque_front = reste & ~ndimage.binary_erosion(reste, structure=croix, border_value=1)
    frontiere = [(int(i), int(j)) for (i, j) in np.argwhere(masque_front)]
    # ANCIEN :
    # frontiere = []
    # for (i,j) in forme_set:
    #     for m in mouvements_simples :
    #         ni, nj = i + m[0], j + m[1]
    #         if (ni, nj) not in forme_set and 0 <= ni < height and 0 <= nj < width:
    #             frontiere.append((i,j))
    #             break

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
                    gradi, gradj = sobel(gris, a, b)   # [LAB] 'gris' est maintenant la luminance L
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
    # [OPT 5] moyenne de "confiance" sur la fenêtre 13x13 pour TOUS les pixels d'un coup (uniform_filter).
    #         Hors image = 0, comme dans ta boucle (qui divisait par taille_patch**2 en ignorant les pixels hors image).
    carte_conf = ndimage.uniform_filter(confiance, size=taille_patch, mode='constant', cval=0.0)
    cval = [carte_conf[i, j] for (i, j) in frontiere]
    # ANCIEN :
    # cval = []
    # for (i,j) in frontiere:
    #     somme = 0
    #     aire = 0
    #     for k in range(-h, h + 1):
    #         for l in range(-h, h + 1):
    #             a, b = i + k, j + l
    #             if 0 <= a < height and 0 <= b < width:
    #                 aire += 1
    #                 somme += confiance[a, b]
    #     cval.append(somme / (taille_patch**2))

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


    # -----------------------------------------------------------------
    # [CORRIGÉ][MASQUE BINAIRE]  POURQUOI UN MASQUE 0/1 POUR LA SSD ?
    #
    #  On veut mesurer la ressemblance entre le patch cible P (rempli seulement en partie) et un patch source
    #  I(q), en ne regardant QUE les pixels déjà connus de P :
    #        SSD(q) = somme, sur les pixels x du patch, de  M(x) * ( P(x) - I(q+x) )^2
    #  où M(x) = 1 si le pixel x de P est connu, 0 sinon.
    #
    #  ANCIEN CODE : M valait la CONFIANCE (nombre entre 0 et 1 : 1 pour un pixel d'origine, par exemple 0.3 pour
    #  un pixel rempli plus tôt, 0 pour un pixel pas encore rempli). Deux problèmes :
    #
    #  (1) CE N'EST PAS CE QUE FAIT L'ARTICLE. Dans l'article, la confiance sert UNIQUEMENT à choisir l'ORDRE de
    #      remplissage (priorité P = C * D). Une fois qu'un pixel est rempli, sa couleur est une donnée comme une
    #      autre et compte autant qu'un pixel d'origine dans la SSD. Avec la confiance comme poids, un pixel
    #      synthétisé (confiance 0.2) pèse 5 fois moins qu'un pixel d'origine : le meilleur patch ressemble surtout
    #      aux pixels d'origine et ignore ce qu'on vient de synthétiser juste à côté -> moins de continuité d'un
    #      patch au suivant (jointures visibles, structures qui se coupent).
    #
    #  (2) LE DÉVELOPPEMENT EN 3 TERMES N'EST EXACT QU'AVEC UN MASQUE 0/1. On calcule la SSD par FFT en la
    #      développant (a-b)^2 = a^2 - 2ab + b^2 :
    #        SSD(q) =  somme M*P^2   -  2 * somme M*P*I(q+x)   +   somme M*I(q+x)^2
    #                   (term3)            (term2)                   (term1)
    #      term1 = correlate(I^2, M)      -> somme M*I^2        : juste pour n'importe quel M
    #      term2 = -2*correlate(I, M*P)   -> -2 somme M*P*I     : juste pour n'importe quel M
    #      term3 = somme (M*P)^2          -> vaut somme M*P^2 SEULEMENT si M^2 = M, c'est-à-dire M = 0 ou 1.
    #      Avec M = 0.3 on obtient 0.09*P^2 au lieu de 0.3*P^2. Comme term3 ne dépend pas de q, ça décale tous les
    #      scores de la même quantité : le "meilleur q" ne change pas À CAUSE DE ÇA. Mais le nombre obtenu n'est
    #      plus une vraie SSD (il peut être négatif), donc inutilisable comme mesure d'erreur (normalisation en MSE,
    #      comparaison entre patchs...). Le vrai problème de fond reste le (1) : la pondération elle-même.
    #
    #  CORRECTION : M = 'connu' (0/1). Tous les pixels déjà remplis comptent pareil et les 3 termes redonnent
    #  exactement la SSD. Les pixels du trou pas encore remplis (M = 0) sont ignorés : c'est indispensable, car
    #  dans 'img' ils contiennent encore les couleurs de l'objet à effacer.
    #  [LAB] Rien ne change dans la formule : la SSD somme simplement les carrés des écarts des 3 canaux L, a, b.
    # -----------------------------------------------------------------
    zone_inconnue = np.zeros((taille_patch, taille_patch))
    zone_inconnue[pi_min:pi_max, pj_min:pj_max] = connu[i_min:i_max, j_min:j_max]   # ANCIEN : confiance[i_min:i_max, j_min:j_max]

    patch_a_changer = np.zeros((taille_patch, taille_patch, nb_canaux))

    for c in range(nb_canaux):

        patch_a_changer[pi_min:pi_max, pj_min:pj_max, c] = img[i_min:i_max, j_min:j_max, c] * zone_inconnue[pi_min:pi_max, pj_min:pj_max]

    term3 = np.sum(patch_a_changer**2)

    # [AJOUT: BANDE] la carte SSD n'est calculée que sur la zone utile [r0:r1, c0:c1] (img_zone est une VUE de img, toujours à jour)
    img_zone = img[r0:r1, c0:c1]
    ssd_map = np.zeros((r1 - r0, c1 - c0))     # ANCIEN : np.zeros((height, width))
    for c in range(nb_canaux):
        im_canal_c = img_zone[:, :, c]         # ANCIEN : img[:, :, c]   ([LAB] canal c = L, a ou b)
        term1 = correlate(im_canal_c**2, zone_inconnue, mode='same', method='fft')
        term2 = -2 * correlate(im_canal_c, patch_a_changer[:, :, c], mode='same', method='fft')
        ssd_map += term1 + term2

    ssd_map += term3

    # [OPT 4] [AJOUT: BANDE] [AJOUT: DISTANCE] choix du meilleur patch (voir la fonction meilleure_source plus haut)
    q, _ = meilleure_source(ssd_map, zone_inconnue.sum(), i_prio, j_prio)
    # ANCIEN :
    # meilleur_ssd = float('inf')
    # q = None
    # for (i, j) in sources:
    #     ssd = round(ssd_map[i, j], 2)
    #     if ssd < meilleur_ssd:
    #         meilleur_ssd = ssd
    #         q = (i, j)

    for i in range(-h, h + 1):
        for j in range(-h, h + 1):
            a, b = i_prio + i, j_prio + j
            if 0 <= a < height and 0 <= b < width and (a,b) in forme_set:
                for c in range(nb_canaux):
                    img[a, b, c] = img[q[0] + i, q[1] + j, c]
                # [LAB] on met à jour 'gris' avec la luminance L du pixel copié.
                #   ANCIEN : s = somme des canaux ; gris[a, b] = s / nb_canaux  (moyenne R,G,B, qui n'a plus de sens en Lab)
                gris[a, b] = img[a, b, 0]
                confiance[a, b] = C_chapeau
                connu[a, b] = 1  # [CORRIGÉ] le pixel devient "connu" pour le SSD
                reste[a, b] = False  # [OPT 5] on garde le masque numpy "reste" synchronisé avec forme_set
                forme_set.remove((a,b))

    print("pixels pas encore remplis", len(forme_set))
#méthode itérative
vote = np.zeros_like(img)
poid = np.zeros_like(img)
s=6
img_trou_ech = [forme[i] for i in range(len(forme)) if (forme[i][0] % s == 0 and forme[i][1] % s == 0)]
print("gogo")
nb_i=5
mediane = False   # [AJOUT: WEXLER] False = vote pondéré à la Wexler (nouveau défaut) ; True = médiane (ton ancien mode)
for k in range(nb_i):
    # [LAB] ANCIEN : skio.imsave(f"resultat_{k}.png", np.clip(img, 0, 255).astype(np.uint8))
    #   img est en Lab : on le reconvertit en RGB avant d'enregistrer.
    skio.imsave(f"resultat_{k}.png", vers_rgb_uint8(img))
    votes_mediane = {}
    # [CORRIGÉ] on remet les accumulateurs à zéro à chaque itération (avant ils gardaient les votes des itérations précédentes)
    vote = np.zeros_like(img)
    poid = np.zeros((height, width))   # [AJOUT: WEXLER] 2D : le poids est le même pour les 3 canaux (ANCIEN : zeros_like(img))
    liste_votes = []  # [AJOUT: WEXLER] on stocke tous les votes puis on pondère une fois σ connu
    img_zone = img[r0:r1, c0:c1]   # [AJOUT: BANDE] vue de img sur la zone utile (à jour au début de chaque itération)
    for (i, j) in img_trou_ech:
        i_min, i_max = max(0, i - h), min(height, i + h + 1)
        j_min, j_max = max(0, j - h), min(width, j + h + 1)

        # [CORRIGÉ] le patch et son masque sont posés dans un cadre 13x13 centré sur (i,j) (comme dans la phase 1).
        #   Avant, près des bords de l'image le noyau était tronqué (taille paire possible) et la corrélation 'same'
        #   décalait le patch d'un demi-pixel. ANCIEN :
        #   patch = img[i_min:i_max, j_min:j_max].copy() ; mask = np.ones((i_max - i_min, j_max - j_min))
        pi_min, pi_max = h - (i - i_min), h + (i_max - i)
        pj_min, pj_max = h - (j - j_min), h + (j_max - j)
        mask = np.zeros((taille_patch, taille_patch))     # [MASQUE BINAIRE] ici le masque est bien 0/1 (1 = pixel dans l'image)
        mask[pi_min:pi_max, pj_min:pj_max] = 1.0
        patch = np.zeros((taille_patch, taille_patch, nb_canaux))
        patch[pi_min:pi_max, pj_min:pj_max] = img[i_min:i_max, j_min:j_max]
        ssd_map = np.zeros((r1 - r0, c1 - c0))     # [AJOUT: BANDE] ANCIEN : np.zeros((height, width))
        term3 = np.sum(patch ** 2)

        for c in range(nb_canaux):
            im_canal = img_zone[:, :, c]           # [AJOUT: BANDE] ANCIEN : img[:, :, c]
            term1 = correlate(im_canal**2, mask, mode='same', method='fft')
            term2 = -2 * correlate(im_canal, patch[:, :, c], mode='same', method='fft')
            ssd_map += term1 + term2
        ssd_map += term3

        # [OPT 4] [AJOUT: BANDE] [AJOUT: DISTANCE] argmin numpy + bande + pénalité (voir meilleure_source)
        (qi, qj), d_mse = meilleure_source(ssd_map, mask.sum(), i, j)
        # ANCIEN :
        # meilleur_ssd = float('inf')
        # q = None
        # temperature_l=[]
        # for (si, sj) in sources:
        #     ssd = ssd_map[si, sj]
        #     if ssd < meilleur_ssd:
        #         meilleur_ssd = ssd
        #         qi,qj = (si, sj)
        #         temperature_l.append(meilleur_ssd)
        #     temperature = np.mean(np.array( temperature_l)) - 3*np.std(np.array(temperature_l))

        di_haut, di_bas = i - i_min, i_max - i
        dj_gauche, dj_droite = j - j_min, j_max - j

        patch_source = img[qi - di_haut : qi + di_bas, qj - dj_gauche : qj + dj_droite]

        if mediane == False :
            # [AJOUT: WEXLER] on stocke le vote (avec son MSE) ; il sera pondéré une fois σ connu
            liste_votes.append((i_min, i_max, j_min, j_max, patch_source.copy(), d_mse))

        else:
            for d_i in range(patch_source.shape[0]):
                for d_j in range(patch_source.shape[1]):
                    pi, pj = i_min + d_i, j_min + d_j
                    if (pi, pj) not in votes_mediane:
                        votes_mediane[(pi, pj)] = []
                    votes_mediane[(pi, pj)].append(patch_source[d_i, d_j])

    if mediane == False:
        # [AJOUT: WEXLER] vote pondéré : chaque patch vote avec le poids  w = exp(-d / (2 σ²))
        #   d  = MSE entre le patch cible et son meilleur patch source (petit = bonne correspondance = gros poids)
        #        [LAB] d est maintenant en unités Lab^2 : une différence de 1 correspond environ à la plus petite
        #        différence de couleur perceptible, ce qui rend les poids plus cohérents qu'en RGB.
        #   σ² = 75e percentile des d de l'itération (convention de Wexler et al. : on s'adapte à l'échelle des erreurs)
        #   Chaque pixel du trou devient la moyenne pondérée de tous les votes qu'il reçoit (moyenne faite en Lab).
        if len(liste_votes) > 0:
            d_tous = np.array([v[5] for v in liste_votes])
            sigma2 = np.percentile(d_tous, 75) + 1e-8
            for (i_min, i_max, j_min, j_max, patch_source, d_v) in liste_votes:
                w = np.exp(-d_v / (2 * sigma2))
                vote[i_min:i_max, j_min:j_max] += patch_source * w
                poid[i_min:i_max, j_min:j_max] += w
            # [CORRIGÉ] on ne modifie que les pixels du trou d'origine
            zone_maj = (poid > 0) & trou
            img[zone_maj] = vote[zone_maj] / poid[zone_maj][:, None]
    else:
        for (pi, pj), liste_valeurs in votes_mediane.items():
            if trou[pi, pj]:  # [CORRIGÉ] avant, les pixels connus hors du trou étaient aussi écrasés par la médiane
                img[pi, pj] = np.median(liste_valeurs, axis=0)   # [LAB] médiane canal par canal (L, a, b)
    print("ca roule")
# [LAB] ANCIEN : skio.imsave("resultat.png", np.clip(img, 0, 255).astype(np.uint8))
#   Conversion finale Lab -> RGB pour l'enregistrement.
skio.imsave("resultat.png", vers_rgb_uint8(img))
print("résultat enregistré dans resultat.png")