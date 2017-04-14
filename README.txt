Le code du projet est relativement long, s�par� en plusieurs fichiers selon les diff�rentes grandes parties de
l'algorithme. Par manque de temps, le projet n'est pas pleinement termin�. Plus particuli�rement :

-> L'initialisation fonctionne � peu pr�s convenablement. On trouvera en particulier plusieurs tests dans le
main.cpp. Notez que les param�tres du projet peuvent �tre modifi�s depuis le fichier image.h

-> L'optimisation fonctionne plus ou moins. En particulier, l'article est assez lacunaire sur cette optimisation,
et de fait le calcul du gradient est non trivial, et des non lin�arit�s apparaissent avec les contraintes. Pour
contourner ce probl�me, j'ai utilis� une fonction d'optimisation diff�rentes. Toutefois, celle-ci a l'inconv�nient
majeur de faire d�border la m�moire (unsufficient memory) sur des images de trop haute r�solution, ce pour une
raison qui m'�chappe. L'optimisation fonctionne toutefois sur des images de (tr�s) petites tailles. Pour faire
tourner le code, penser � passer RESIZE � true dans image.h et � choisir un ratio assez petit.