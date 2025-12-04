
# traduction du montage de geogebra compas v2

a = controle de l'unique DOF

- fixed : A, B, C, J

- ROT(C, A, a) => C'
-  variable : a
-  C' = rotation de C autour de A d'un angle a => C'

- JOINT(C', J, u, s) => K
-  param u, s
-  dist(C' K) == u et  dist(J, K) == s => solve K

- PROJ(J, K, v) => N
-  param v
-  N sur ligne (J K) et d(J,N)== v => N

- JOINT(C', B, o, b) => F
-  param o, b
-  dist(C', F) == o et dist(B, F) == b => F

- PROJ(C', F, q + dist(C',F)) => H 
-  param q
-  dist(F, H) == q et H sur la ligne (F C') => H

- JOINT(H, N, n, w) => O
- param n w
- dist(H, O) == n et dist(N, O) == w => O


tracer segment(H, O)

#
approche :

## code forward

  obtenir le code qui calcule les terminaux H et O en fct de a
  suppose de savoir résoudre les differentes étapes
  il y en a de 2 types :
  - ROT, PROJ : calcul direct
  - JOINT : demande resolution de l'équation
  dans les 2 types on sait coder le forward

on obtient le code pytorch de ça

## optim

collecter les variables : tous les params et les 4 points fixes donnés
sauf A et B qu'on fixe
(A B) étant vertical

ensuite faire varier a dans l'intervalle permis , eg [ 0, 180°]

calculer les coords de tous les points
appliquer les contraintes : calculer le coût
obtenir un loss
ajuster les variables

# contraintes

obliger H et O à descendre verticalement sur un distance d1 au début de l'intervalle
forcer la distance (H O) = d2

éviter collisions des axes ( épaisseurs des boulons)

introduire une épaisseur des bras
idée : calculer la surface des bras, marquer l'espace total balayé par les bras et minimiser sa surface
pénaliser le recouvrement d'un axe avec un bras auquel il n'est pas connecté
introduire un min/max sur la longueur des bras  

forcer (H O) à arriver en un certain point à la fin

ou

forcer (H O) à être horizontaux au début
forcer (H O) à être verticaux à la fin
forcer (H O) à être alignés en dessous de (AB)




