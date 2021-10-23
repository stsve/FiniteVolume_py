

"""
Reconstruction du gradient aux triangles par least-squares
INPUT: 
xx, yy : array de coordonnées des noeuds du maillage
tri    : connectivité des triangles
are    : tableau des arêtes
bcdata : tableau des conditions aux limites
soltri : champ aux triangles 
    
OUTPUT: 
GRADPHI = gradient aux triangles (ntri,2) (gradx,grady)

Version 1.0 Jean-Yves Trépanier - 7 février 2020
"""
def TriGradientLS(xx,yy,tri,are,bcdata,soltri):
    """ Calcul du gradient aux triangles par least-squares """

    xpc,ypc = TriCenter(xx,yy,tri)
    naref  = np.size(np.nonzero(are[:,3]==-1))    #nombre d'aretes en frontière
    naretes = are.shape[0]
    ntri    = tri.shape[0]
    #les métriques des aretes 
    DAI,DXI,PNXI,PXIET,NAx,NAy,DX,DY = AreMetric(xx,yy,tri,are)   
   
    #construction des matrices 2x2 par triangle pour le least-square
    ATA  = np.zeros((ntri,2,2))       #matrice LS par triangle
    ATAI = np.zeros((ntri,2,2))       #matrice LS inverse par triangle
    ALS  = np.zeros((2,2))            #matrice LS locale
    ALSI = np.zeros((2,2))            #matrice LS locale inverse
    BLS  = np.zeros(2)                #membre de droite local
    ATB  = np.zeros((ntri,2))         #membre de droite par triangle
    GRADPHI = np.zeros((ntri,2))      #gradient par triangle
    PHI = soltri                      #solution aux triangles à reconstruire
    
    #aretes internes
    for iare in range(naref,naretes):
        ALS[0,0]=DX[iare]*DX[iare]
        ALS[1,0]=DX[iare]*DY[iare]
        ALS[0,1]=DY[iare]*DX[iare]
        ALS[1,1]=DY[iare]*DY[iare]
        
        ATA[are[iare,2]]+=ALS
        ATA[are[iare,3]]+=ALS
        
    #aretes frontieres
    for iare in range(0,naref):
        if ( bcdata[are[iare,4]][0] == 'Dirichlet' ):
            ALS[0,0]=DX[iare]*DX[iare]
            ALS[1,0]=DX[iare]*DY[iare]
            ALS[0,1]=DY[iare]*DX[iare]
            ALS[1,1]=DY[iare]*DY[iare]
            
            ATA[are[iare,2]]+=ALS
            
        if ( bcdata[are[iare,4]][0] == 'Neumann' ):
            ALS[0,0]=DX[iare]*DX[iare]*NAx[iare]*NAx[iare]
            ALS[1,0]=DX[iare]*DY[iare]*NAx[iare]*NAy[iare]
            ALS[0,1]=DY[iare]*DX[iare]*NAy[iare]*NAx[iare]
            ALS[1,1]=DY[iare]*DY[iare]*NAy[iare]*NAy[iare]
            
            ATA[are[iare,2]]+=ALS
        
        
    #inversion des matrices de least-square 2x2
    for itri in range(0,ntri):
        ALS = ATA[itri]
        ALSI = np.linalg.inv(ALS)
        ATAI[itri]=ALSI
    
    #calcul du gradient par triangle par least-square
    #contruction du membre de droite du système Least-square   
    #aretes internes
    for iare in range(naref,naretes):
        BLS[0]=DX[iare]*(PHI[are[iare,3]]-PHI[are[iare,2]])
        BLS[1]=DY[iare]*(PHI[are[iare,3]]-PHI[are[iare,2]])
        
        ATB[are[iare,2]]+=BLS
        ATB[are[iare,3]]+=BLS
        
    #aretes frontieres
    for iare in range(0,naref):
        if ( bcdata[are[iare,4]][0] == 'Dirichlet' ):
            #récupere la valeur de la CL dirichlet
            if(callable(bcdata[are[iare,4]][1])):
                Xa = xx[are[iare,0]] 
                Xb = xx[are[iare,1]]
                Ya = yy[are[iare,0]] 
                Yb = yy[are[iare,1]]
                XA = (Xa + Xb) / 2.0
                YA = (Ya + Yb) / 2.0
                bcvalue = bcdata[are[iare,4]][1](XA,YA)    
            else:
                bcvalue = bcdata[are[iare,4]][1]
        
            BLS[0]=DX[iare]*(bcvalue-PHI[are[iare,2]])
            BLS[1]=DY[iare]*(bcvalue-PHI[are[iare,2]])
            
            ATB[are[iare,2]]+=BLS   
    
    for itri in range(0,ntri):
        
#        print('itri, ATA(itri)  ',ATA[itri])
#        print('itri, ATB(itri)  ',ATB[itri])
        
        GRADPHI[itri]=np.dot(ATAI[itri],ATB[itri])
    
    return GRADPHI
