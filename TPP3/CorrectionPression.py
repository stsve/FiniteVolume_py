"""
Calcul de la correction de la pression et des vitesses débitantes pour l'algorithme SIMPLE
INPUT: 
xx, yy : array de coordonnées des noeuds du maillage
tri    : connectivité des triangles
are    : tableau des arêtes
bcdpc  : Tableau des conditions limites pour la correction de pression
         (seul le type 'Sortie' est important, où la pression est imposée)
AS     : matrice du système de convection-diffusion pour u (Sparse)
UF     : champ de vitesse aux faces en sortie de Rhie-Chow 
          (positif si la vitesse sort du triangles de gauche)  
OUTPUT: 
PPrime : Correction de la pression aux triangles
UfNew  : Vitesse débitante corrigée (devrait être à divergence nulle)

Version 1.0 Jean-Yves Trépanier - 14 février 2020
"""
def TriCorrectionPression(xx,yy,tri,are,bcdp,AS,UF):
    """ Calcul du correction de la pression pour l'algorithme SIMPLE """

    ntri = tri.shape[0]
    naref  = np.size(np.nonzero(are[:,3]==-1))    #nombre d'aretes en frontière
    naretes = are.shape[0]
    DAI,DXI,PNXI,PXIET,NAx,NAy,DX,DY = AreMetric(xx,yy,tri,are)   

    DV = TriArea(xx,yy,tri)    #aire des triangles    
    DAu = AS.diagonal()        #diagonale de la matrice de momentum    
    
    MP = np.zeros((ntri,ntri))
    DFI = np.zeros(naretes)
    UfNew = np.zeros(naretes)
    UfNew += UF                 #utile pour entrée et paroi solide
    
    #aretes internes
    for iare in range(naref,naretes):
        
        DFI[iare] = 0.5* ( DV[are[iare,2]]/DAu[are[iare,2]] +              \
                           DV[are[iare,3]]/DAu[are[iare,3]] ) /DXI[iare]
        
        MP[are[iare,2],are[iare,2]]+=DFI[iare]*DAI[iare]
        MP[are[iare,3],are[iare,3]]+=DFI[iare]*DAI[iare]
        MP[are[iare,2],are[iare,3]] =-DFI[iare]*DAI[iare]
        MP[are[iare,3],are[iare,2]] =-DFI[iare]*DAI[iare]

    #aretes frontieres
    for iare in range(0,naref):
        if ( bcdp[are[iare,4]][0] == 'Sortie' ):

            DFI[iare] = DV[are[iare,2]]/DAu[are[iare,2]]/DXI[iare]
            
            MP[are[iare,2],are[iare,2]]+=DFI[iare]*DAI[iare]
             
    #membre de droite - divergence de la vitesse
    BD = TriDivergence(xx,yy,tri,are,UF)    
   
    # Résolution du système Ax = b     
    #PPrime = np.linalg.solve(MP, BD)

    #Résolution d'un système linéaire sparse méthode directe
    MPS = sps.csr_matrix(MP)
    PPrime = spsolve(MPS,BD)
    
#    print ('correction pression MP,BD ',MP, BD)
    #correction des vitesses débitantes
    #aretes internes
    for iare in range(naref,naretes):
        
        UfNew[iare] = UF[iare] + DFI[iare] *                         \
                ( PPrime[are[iare,2]] - PPrime[are[iare,3]] )
    
    #aretes frontieres
    for iare in range(0,naref):
        if ( bcdp[are[iare,4]][0] == 'Sortie' ):

            UfNew[iare] = UF[iare] + DFI[iare] * PPrime[are[iare,2]]    

    return PPrime, UfNew

