Le code du projet est relativement long, séparé en plusieurs fichiers selon les différentes grandes parties de
l'algorithme. Par manque de temps, le projet n'est pas pleinement terminé. Plus particulièrement :

-> L'initialisation fonctionne à peu près convenablement. On trouvera en particulier plusieurs tests dans le
main.cpp. Notez que les paramètres du projet peuvent être modifiés depuis le fichier image.h

-> L'optimisation fonctionne plus ou moins. En particulier, l'article est assez lacunaire sur cette optimisation,
et de fait le calcul du gradient est non trivial, et des non linéarités apparaissent avec les contraintes. Pour
contourner ce problème, j'ai utilisé une fonction d'optimisation différentes. Toutefois, celle-ci a l'inconvénient
majeur de faire déborder la mémoire (unsufficient memory) sur des images de trop haute résolution, ce pour une
raison qui m'échappe. L'optimisation fonctionne toutefois sur des images de (très) petites tailles. Pour faire
tourner le code, penser à passer RESIZE à true dans image.h et à choisir un ratio assez petit.