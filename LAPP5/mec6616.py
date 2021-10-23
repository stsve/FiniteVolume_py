"""
Module de Fonctions Python pour maillage triangle pour le cours mec6616 
Jean-Yves Trépanier - Eddy Petro - Janvier 2020
"""


"""
FONCTION : RectMesh(xmin,xmax,ymin,ymax,nx,ny)

Maillage d'un rectangle en triangles avec Matplotlib.tri.Triangulation
Jean-Yves Trépanier - Janvier 2020

                             cf=3
    ymax   -----------------------------------------
           |                                       |
    cf=0   |                                       | cf=2
           |                                       |
    ymin   -----------------------------------------
          xmin           cf=1                     xmax

INPUT: 
xmin, xmax, ymin, ymax : coordonnées réelles du rectangle
nx,ny : Nombre de cellules en x et y ()

OUTPUT: 
xx, yy : array de coordonnées des noeuds du maillage
triangles : connectivité des triangles : n0,n1,n2 trigo pour chaque triangle
aretes : structure des arètes : n0,n1,tg,td,cf

          n1 *                 arete de n0 à n1
           / |\                tg : triangle à gauche
      /      |     \           td : triangle à droite 
  /          |         \       si l'arete est sur une frontière cf, alors 
  \  tg      |cf  td  /           tg = -1 et cf = 0,1,2,3 selon la frontière
     \       |      /          pour arete internes, tg et tf sont >=0 et cf=0
         \   |   /             
            \|/                dans le tableau aretes, les aretes frontières 
          n0 *                 sont toujours au début du tableau

"""
def RectMesh(xmin,xmax,ymin,ymax,nx,ny):
    """ Génère un maillage de triangles régulier dans un rectangle"""
    import matplotlib.tri as mtri
    import numpy as np
    
    # taille du rectangle et nombre de cellules en x et y 
    dx = (xmax-xmin)/nx           #taille des intervalles 
    dy = (ymax-ymin)/ny
    
    #points du maillage (nodes)
    xx = np.tile(np.linspace(xmin, xmax, nx+1),ny+1)
    yy = np.repeat(np.linspace(ymin, ymax, ny+1),nx+1)
    
    #triangulation des points
    trirectangle = mtri.Triangulation(xx, yy)
    triangles=trirectangle.triangles

    aretei,aretef = TriAretes(xx,yy,triangles)
    
#identifie les 4 frontières Ouest:0, Sud:1, Est:2, Nord:3    
    for iare in range(0,aretef.shape[0]):
        xmid = 0.5*(xx[aretef[iare,0]]+xx[aretef[iare,1]])
        ymid = 0.5*(yy[aretef[iare,0]]+yy[aretef[iare,1]])
        if(xmid < xmin+0.1*dx):
            aretef[iare,4]=0
        if(ymid < ymin+0.1*dy):
            aretef[iare,4]=1
        if(xmid > xmax-0.1*dx):
            aretef[iare,4]=2
        if(ymid > ymax-0.1*dy):
            aretef[iare,4]=3
    
    aretes = np.append(aretef,aretei,axis=0)
 
    return xx,yy,triangles,aretes

"""
FONCTION : RectGmsh(xmin,xmax,ymin,ymax,lc)

Maillage d'un rectangle en triangles avec gmsh
Eddy Petro et Jean-Yves Trépanier - Janvier 2020

                             cf=3
    ymax   -----------------------------------------
           |                                       |
    cf=0   |                                       | cf=2
           |                                       |
    ymin   -----------------------------------------
          xmin           cf=1                     xmax

INPUT: 
xmin, xmax, ymin, ymax : coordonnées réelles du rectangle
lc : Longueur visée des arètes des triangles

OUTPUT: 
xx, yy : array de coordonnées des noeuds du maillage
triangles : connectivité des triangles : n0,n1,n2 trigo pour chaque triangle
aretes : structure des arètes : n0,n1,tg,td,cf

          n1 *                 arete de n0 à n1
           / |\                tg : triangle à gauche
      /      |     \           td : triangle à droite 
  /          |         \       si l'arete est sur une frontière cf, alors 
  \  tg      |cf  td  /           tg = -1 et cf = 0,1,2,3 selon la frontière
     \       |      /          pour arete internes, tg et tf sont >=0 et cf=0
         \   |   /             
            \|/                dans le tableau aretes, les aretes frontières 
          n0 *                 sont toujours au début du tableau

"""    
def RectGmsh(x1,x2,y1,y2,lc):
    """ Génère un maillage de triangles de taille lc dans un rectangle"""    
    import numpy as np
    import gmsh

    # Before using any functions in the Python API, Gmsh must be initialized.
    gmsh.initialize()
    
    # By default Gmsh will not print out any messages: in order to output messages
    # on the terminal, just set the standard Gmsh option "General.Terminal" (same
    # format and meaning as in .geo files):
    gmsh.option.setNumber("General.Terminal", 1)
    
    # Next we add a new model named "t1" (if gmsh.model.add() is not called a new
    # unnamed model will be created on the fly, if necessary):
    gmsh.model.add("rectangle")
    
    gmsh.model.geo.addPoint(x2,y2,0,lc,1)
    gmsh.model.geo.addPoint(x1,y2,0,lc,2)
    gmsh.model.geo.addPoint(x1,y1,0,lc,3)
    gmsh.model.geo.addPoint(x2,y1,0,lc,4)
     
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    
    gmsh.model.geo.addCurveLoop([1,2,3,4],1)
    
    gmsh.model.geo.addPlaneSurface([1],11)
    
    gmsh.model.addPhysicalGroup(2, [11], 11)
    gmsh.model.setPhysicalName(2, 11, "DOMAINE")
    
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate()
    
#    gmsh.write("rectangle.vtk")
#    gmsh.write("rectangle.msh")
    
    nodeTags, coord, parametricCoord = gmsh.model.mesh.getNodes()
    
    # Reshape to get right format
    points = np.reshape(coord, (int(coord.size / 3), 3))
    
    element_types, element_tags, node_tags = gmsh.model.mesh.getElements()

    # End Generate mesh
    gmsh.finalize()

#Reecrire dans le format voulu    
    Bord, Connectivity, Coin = node_tags
    CT = Connectivity.reshape(int(Connectivity.size/3),3)-1
    
    node=[]
    nodeTaglist = nodeTags.tolist()
    nodenumbersread = []
    for icells in range(0,len(CT)):  
        for i in range(0,3):  
            if (CT[icells][i] in nodenumbersread) == 0:
                nodenumbersread.append(CT[icells][i]) 
                point_index = nodeTaglist.index(CT[icells][i] + 1)
                node = np.append(node, points[point_index,0])
                node = np.append(node, points[point_index,1] )
                node = np.append(node, 0)
    
    node = node.reshape(int(node.size/3),3)
    
    nodesreordered =  np.zeros((len(node), 3))
    for i in range(0,node.shape[0]):
        nodesreordered[nodenumbersread[i]] = node[i]
        
    xx=nodesreordered[:,0]
    yy=nodesreordered[:,1]
    
    aretei,aretef = TriAretes(xx,yy,CT)

    #identifie les 4 frontières Ouest:0, Sud:1, Est:2, Nord:3    
    for iare in range(0,aretef.shape[0]):
        xmid = 0.5*(xx[aretef[iare,0]]+xx[aretef[iare,1]])
        ymid = 0.5*(yy[aretef[iare,0]]+yy[aretef[iare,1]])
        if(xmid < x1+0.1*lc):
            aretef[iare,4]=0
        if(ymid < y1+0.1*lc):
            aretef[iare,4]=1
        if(xmid > x2-0.1*lc):
            aretef[iare,4]=2
        if(ymid > y2-0.1*lc):
            aretef[iare,4]=3

    aretes = np.append(aretef,aretei,axis=0)
    
    return xx,yy,CT,aretes

"""
FONCTION : AirfoilGmsh(filename,d,lc1,lc2)

Maillage autour d'un profil en triangles avec gmsh
Eddy Petro et Jean-Yves Trépanier - Janvier 2020

                             cf=3
    d      -----------------------------------------
           |                                       |
           |              ________                 |
    cf=0   |     y=0     <________>                | cf=2
           |          x=0  cf=4  x=1               |
           |                                       |
           |                                       |
    -d     -----------------------------------------
          -d           cf=1                       d

INPUT:
filename   :  nom du fichier de points contenant le profil    
d          :  hauteur et longueur de la boite délimitant le champ lointain (min = 2)
lc1        :  longueur d'une arrête de triangle sur le profil
lc2        :  longueur d'une arrête de triangle au niveau du champ lointain    

Note sur le fichier filename: 
    
Fichier texte contenant les points du profil dans le sens horaire. Le premier 
point doit être au bord de fuite. Le profil doit être normalisé entre 
les points (0,0) et (1,0)  
    
x1,y1,0.0
x2,y2,0.0
...
xn,yn,0.0

OUTPUT: 
xx, yy : array de coordonnées des noeuds du maillage
triangles : connectivité des triangles : n0,n1,n2 trigo pour chaque triangle
aretes : structure des arètes : n0,n1,tg,td,cf

          n1 *                 arete de n0 à n1
           / |\                tg : triangle à gauche
      /      |     \           td : triangle à droite 
  /          |         \       si l'arete est sur une frontière cf, alors 
  \  tg      |cf  td  /           tg = -1 et cf = 0,1,2,3 selon la frontière
     \       |      /          pour arete internes, tg et tf sont >=0 et cf=0
         \   |   /             
            \|/                dans le tableau aretes, les aretes frontières 
          n0 *                 sont toujours au début du tableau

""" 
#def AirfoilGmsh(filename,d,lc1,lc2):
#    """ Génère un maillage de triangles autour d'un profil """    
#
#    import numpy as np
#    import gmsh
#
#    # Before using any functions in the Python API, Gmsh must be initialized.
#    gmsh.initialize()
#    
#    # By default Gmsh will not print out any messages: in order to output messages
#    # on the terminal, just set the standard Gmsh option "General.Terminal" (same
#    # format and meaning as in .geo files):
#    gmsh.option.setNumber("General.Terminal", 1)
#    
#    # Next we add a new model named "t1" (if gmsh.model.add() is not called a new
#    # unnamed model will be created on the fly, if necessary):
#    gmsh.model.add('airfoil')
#      
#    pts = np.loadtxt(fname = filename)
#    nbpts = len(pts)
#    
#    spline1 = []
#    spline2 = [] 
#    spline3 = [] 
#    spline4 = []  
#    
#    for ipts in range(0,nbpts):
#        p = gmsh.model.geo.addPoint(pts[ipts,0],pts[ipts,1], 0, lc1, ipts+1)
#        if ipts <= (nbpts/4 - 1):
#           spline1.append(p)
#        if ipts > (nbpts/4 - 2) and ipts <= (nbpts/2):
#           spline2.append(p)
#        if ipts > (nbpts/2 - 1) and ipts <= (3*nbpts/4):
#           spline3.append(p)
#        if ipts > (3*nbpts/4 - 1) and ipts <= nbpts:
#           spline4.append(p)   
#      
#    # ajoute le premier point a la fin de la derniere spline afin de fermer la boucle   
#    spline4.append(1)
#            
#    gmsh.model.geo.addSpline(spline1,1)
#    gmsh.model.geo.addSpline(spline2,2)
#    gmsh.model.geo.addSpline(spline3,3)
#    gmsh.model.geo.addSpline(spline4,4)
#    
#    #frontière externe
#    d = max(d,2)     # d est au moins plus grand que 2
#    gmsh.model.geo.addPoint(d,d,0,lc2,1000)
#    gmsh.model.geo.addPoint(-d,d,0,lc2,1001)
#    gmsh.model.geo.addPoint(-d,-d,0,lc2,1002)
#    gmsh.model.geo.addPoint(d,-d,0,lc2,1003)
#    
#    gmsh.model.geo.addLine(1000, 1001, 5)
#    gmsh.model.geo.addLine(1001, 1002, 6)
#    gmsh.model.geo.addLine(1002, 1003, 7)
#    gmsh.model.geo.addLine(1003, 1000, 8)
#    
#    gmsh.model.geo.addCurveLoop([1,2,3,4],1)
#    gmsh.model.geo.addCurveLoop([5,6,7,8],2)
#    
#    gmsh.model.geo.addPlaneSurface([2,1],11)
#    
#    gmsh.model.addPhysicalGroup(2, [11], 11)
#    gmsh.model.setPhysicalName(2, 11, "DOMAINE")
#    
#    gmsh.model.geo.synchronize()
#    gmsh.model.mesh.generate()
#    
#    nodeTags, coord, parametricCoord = gmsh.model.mesh.getNodes()
#    
#    # Reshape to get right format
#    points = np.reshape(coord, (int(coord.size / 3), 3))
#    
#    element_types, element_tags, node_tags = gmsh.model.mesh.getElements()
#    
#    Bord, Connectivity, Coin = node_tags
#    CT = Connectivity.reshape(int(Connectivity.size/3),3)-1
#      
#    node= []
#    nodeTaglist = nodeTags.tolist()
#    nodenumbersread = []
#    for icells in range(0,len(CT)):  
#        for i in range(0,3):  
#            if (CT[icells][i] in nodenumbersread) == 0:
#                nodenumbersread.append(CT[icells][i]) 
#                point_index = nodeTaglist.index(CT[icells][i] + 1)
#                node = np.append(node, points[point_index,0])
#                node = np.append(node, points[point_index,1] )
#                node = np.append(node, 0)
#    
#    node = node.reshape(int(node.size/3),3)
#    
#    nodesreordered =  np.zeros((len(node), 3))
#    for i in range(0,node.shape[0]):
#        nodesreordered[nodenumbersread[i]] = node[i]
#        
#    xx=nodesreordered[:,0]
#    yy=nodesreordered[:,1]
#     
#    aretei,aretef = TriAretes(xx,yy,CT)
#
#    #identifie les 5 frontières Ouest:0, Sud:1, Est:2, Nord:3, Profil:4   
#    for iare in range(0,aretef.shape[0]):
#        xmid = 0.5*(xx[aretef[iare,0]]+xx[aretef[iare,1]])
#        ymid = 0.5*(yy[aretef[iare,0]]+yy[aretef[iare,1]])
#        if(xmid < -d+0.1*lc2):
#            aretef[iare,4]=0
#        elif(ymid < -d+0.1*lc2):
#            aretef[iare,4]=1
#        elif(xmid > d-0.1*lc2):
#            aretef[iare,4]=2
#        elif(ymid > d-0.1*lc2):
#            aretef[iare,4]=3
#        else:
#            aretef[iare,4]=4
# 
#    aretes = np.append(aretef,aretei,axis=0)   
#        
#    # End Generate mesh
#    gmsh.finalize()
#    
#    return xx, yy, CT, aretes

"""
FONCTION : TriMoyenne(xx,yy,tri,aretes,bcdata)

Exemple de solveur simple : moyenne des triangles voisins
Jean-Yves Trépanier - Janvier 2020

INPUT: 
xx, yy : array coordonnées des noeuds du maillage
tri  : array connectivité des triangles  
aretes : array structure des arètes : n0,n1,tg,td,cf
bcdata :     #codage des conditions limites - tuple de liste
   ex : bcdata = (['Dirichlet',1],['Dirichlet',10],['Dirichlet',5],['Neumann',0]

   note: seulement Neumann=0 est implémenté
    
OUTPUT: 
X : Array Solution pour chaque triangle

"""
def TriMoyenne(xx,yy,tri,aretes,bcdata):
    """ Resout un problème de moyenne aux triangles """    
    import scipy.sparse as sps
    from scipy.sparse.linalg import spsolve
    import numpy as np
    
    ntri = tri.shape[0]
    #remplissage de la matrice et des conditions limites
    A = np.zeros((ntri,ntri))
    B = np.zeros(ntri)
    
    naref = np.size(np.nonzero(aretes[:,3]==-1))    #nombre d'aretes en frontière
    naretes = aretes.shape[0]

#   Aretes en frontière : assemblage de la matrice et membre de droite
    for iare in range(0,naref):
        if ( bcdata[aretes[iare,4]][0] == 'Dirichlet' ) :
            A[aretes[iare,2],aretes[iare,2]]+=2
            B[aretes[iare,2]]+=2*bcdata[aretes[iare,4]][1]
            
#   Aretes internes : assemblage de la matrice
    for iare in range(naref,naretes):
        A[aretes[iare,3],aretes[iare,3]]+=1
        A[aretes[iare,2],aretes[iare,2]]+=1
        A[aretes[iare,2],aretes[iare,3]]=-1
        A[aretes[iare,3],aretes[iare,2]]=-1
    
    #solution d'un système linéaire sparse méthode directe
    AS = sps.csr_matrix(A)
    X = spsolve(AS,B)

    return X

"""
FONCTION : TriAretes(xx,yy,triangles)

Construit les listes des arêtes internes et frontières d'une triangulation

INPUT: 
xx, yy : array de coordonnées des noeuds du maillage
triangles : connectivité des triangles : n0,n1,n2 trigo pour chaque triangle

OUTPUT: 
aretes : structure des arètes : n0,n1,tg,td,cf

          n1 *                 arete de n0 à n1
           / |\                tg : triangle à gauche
      /      |     \           td : triangle à droite 
  /          |         \       si l'arete est sur une frontière cf, alors 
  \  tg      |cf  td  /           tg = -1 et cf = 0,1,2,3 selon la frontière
     \       |      /          pour arete internes, tg et tf sont >=0 et cf=0
         \   |   /             
            \|/                dans le tableau aretes, les aretes frontières 
          n0 *                 sont toujours au début du tableau

"""    
def TriAretes(xx, yy, triangles):
    """ Construit la liste des arêtes internes et frontières """    
    import matplotlib.tri as tri
    import numpy as np

    trian = tri.Triangulation(xx,yy,triangles)
    triangles = trian.triangles
    ntri = triangles.shape[0]
    voisins = trian.neighbors
    
    aretei  = np.zeros((0,5),dtype=int)     # aretes internes
    aretef  = np.zeros((0,5),dtype=int)     # aretes frontières
    for itri in range(0,ntri):
        nod = triangles[itri,:]    # trois noeuds locaux
        voi = voisins[itri,:]      # trois voisins (0 entre les noeuds 0-1)
        for inod in range(0,3):
            if voi[inod] > itri :         #on crée une arête interne
                arete = np.asarray([[nod[(inod)] ,nod[(inod+1)%3], \
                                itri, voi[inod], 0]])
                aretei=np.append(aretei,arete,axis=0)
            if voi[inod] < 0 :         #on crée une arête frontière
                arete = np.asarray([[nod[(inod)] ,nod[(inod+1)%3], \
                                itri, -1, voi[inod] ]])
                aretef=np.append(aretef,arete,axis=0)        

    return aretei,aretef

"""
FONCTION : BackstepGmsh(H1,H2,L1,L2,lc)

Maillage d'une marche descendante en triangles avec gmsh
Eddy Petro et Jean-Yves Trépanier - Janvier 2020

                             
                     cf=3           L1
           -----------------------------------------
    cf=0   |                                       |
       H1  |                                       |   
           |                                       |  H2   
     (0,0) --------------------                    | cf=2
                   L2  cf=1   | cf=1               |
                              ----------------------
                                    cf=1

INPUT: 
H1,H2,L1,L2 : Dimensions caractéristiques de la marche descendante
lc : Longueur visée des arètes des triangles

OUTPUT: 
xx, yy : array de coordonnées des noeuds du maillage
triangles : connectivité des triangles : n0,n1,n2 trigo pour chaque triangle
aretes : structure des arètes : n0,n1,tg,td,cf

          n1 *                 arete de n0 à n1
           / |\                tg : triangle à gauche
      /      |     \           td : triangle à droite 
  /          |         \       si l'arete est sur une frontière cf, alors 
  \  tg      |cf  td  /           tg = -1 et cf = 0,1,2,3 selon la frontière
     \       |      /          pour arete internes, tg et tf sont >=0 et cf=0
         \   |   /             
            \|/                dans le tableau aretes, les aretes frontières 
          n0 *                 sont toujours au début du tableau

"""  

#def BackstepGmsh(H1,H2,L1,L2,lc):
#    
#    import numpy as np
#    import gmsh
#
#    # Before using any functions in the Python API, Gmsh must be initialized.
#    gmsh.initialize()
#    
#    # By default Gmsh will not print out any messages: in order to output messages
#    # on the terminal, just set the standard Gmsh option "General.Terminal" (same
#    # format and meaning as in .geo files):
#    gmsh.option.setNumber("General.Terminal", 1)
#    
#    # Next we add a new model named "t1" (if gmsh.model.add() is not called a new
#    # unnamed model will be created on the fly, if necessary):
#    gmsh.model.add("backstep")
#      
#    gmsh.model.geo.addPoint(0, H1, 0, lc, 1)
#    gmsh.model.geo.addPoint(L1, H1, 0, lc, 2)
#    gmsh.model.geo.addPoint(L1, H1-H2, 0, lc, 3)
#    gmsh.model.geo.addPoint(L2, H1-H2, 0, lc, 4 )
#    gmsh.model.geo.addPoint(L2,0, 0, lc, 5 )
#    gmsh.model.geo.addPoint(0, 0, 0, lc, 6)
#        
#    gmsh.model.geo.addLine(1, 2, 1)
#    gmsh.model.geo.addLine(2, 3, 2)
#    gmsh.model.geo.addLine(3, 4, 3)
#    gmsh.model.geo.addLine(4, 5, 4)
#    gmsh.model.geo.addLine(5, 6, 5)
#    gmsh.model.geo.addLine(6, 1, 6)
#    
#    gmsh.model.geo.addCurveLoop([1,2,3,4,5,6],1)
#    
#    gmsh.model.geo.addPlaneSurface([1],11)
#    
#    gmsh.model.addPhysicalGroup(2, [11], 11)
#    gmsh.model.setPhysicalName(2, 11, "DOMAINE")
#    
#    gmsh.model.geo.synchronize()
#    gmsh.model.mesh.generate()
#    
#    nodeTags, coord, parametricCoord = gmsh.model.mesh.getNodes()
#    
#    # Reshape to get right format
#    points = np.reshape(coord, (int(coord.size / 3), 3))
#    
#    element_types, element_tags, node_tags = gmsh.model.mesh.getElements()
#    
#    Bord, Connectivity, Coin = node_tags
#    CT = Connectivity.reshape(int(Connectivity.size/3),3)-1
#       
#    node=[]
#    nodeTaglist = nodeTags.tolist()
#    nodenumbersread = []
#    for icells in range(0,len(CT)):  
#        for i in range(0,3):  
#            if (CT[icells][i] in nodenumbersread) == 0:
#                nodenumbersread.append(CT[icells][i]) 
#                point_index = nodeTaglist.index(CT[icells][i] + 1)
#                node = np.append(node, points[point_index,0])
#                node = np.append(node, points[point_index,1] )
#                node = np.append(node, 0)
#    
#    node = node.reshape(int(node.size/3),3)
#    
#    nodesreordered =  np.zeros((len(node), 3))
#    for i in range(0,node.shape[0]):
#        nodesreordered[nodenumbersread[i]] = node[i]
#     
#    # End Generate mesh
#    gmsh.finalize()    
#        
#    xx=nodesreordered[:,0]
#    yy=nodesreordered[:,1]
#    
#    aretei,aretef = TriAretes(xx,yy,CT)    
#
#    #identifie les 4 frontières Ouest:0, Sud:1, Est:2, Nord:3  
#    for iare in range(0,aretef.shape[0]):
#        xmid = 0.5*(xx[aretef[iare,0]]+xx[aretef[iare,1]])
#        ymid = 0.5*(yy[aretef[iare,0]]+yy[aretef[iare,1]])
#        if(xmid < 0.1*lc):
#            aretef[iare,4]=0
#        elif(xmid > L1-0.1*lc):
#            aretef[iare,4]=2
#        elif(ymid > H1-0.1*lc):
#            aretef[iare,4]=3
#        else:
#            aretef[iare,4]=1
#    
#    aretes = np.append(aretef,aretei,axis=0)
#           
#    return xx,yy, CT, aretes

"""
FONCTION :  CircleGmsh(xmin,ymin,xmax,ymax,R,lc,lc2)

Maillage d'un cercle dans un rectangle en triangles avec gmsh
Eddy Petro et Jean-Yves Trépanier - Janvier 2020

                             cf=3
    ymax=y2-----------------------------------------
           |                                       |
    cf=0   |                 .-.                   |  cf=2
           |                 '-'cf=4               | 
           |                                       |
    ymin=y1-----------------------------------------
          xmin=x1           cf=1                   xmax=x2

INPUT: 
xmin, xmax, ymin, ymax : coordonnées réelles du rectangle
R   : Rayon du cercle
lc  : Longueur visée des arètes des triangles appliqués au côtés du rectangle
lc2 : Longueur visée des arètes des triangles appliqués au cercle

OUTPUT: 
xx, yy : array de coordonnées des noeuds du maillage
triangles : connectivité des triangles : n0,n1,n2 trigo pour chaque triangle
aretes : structure des arètes : n0,n1,tg,td,cf

          n1 *                 arete de n0 à n1
           / |\                tg : triangle à gauche
      /      |     \           td : triangle à droite 
  /          |         \       si l'arete est sur une frontière cf, alors 
  \  tg      |cf  td  /           tg = -1 et cf = 0,1,2,3 selon la frontière
     \       |      /          pour arete internes, tg et tf sont >=0 et cf=0
         \   |   /             
            \|/                dans le tableau aretes, les aretes frontières 
          n0 *                 sont toujours au début du tableau

"""    

#def CircleGmsh(x1,x2,y1,y2,R,lc,lc2):
#
#    import numpy as np
#    import gmsh
#
#    # Before using any functions in the Python API, Gmsh must be initialized.
#    gmsh.initialize()
#    
#    # By default Gmsh will not print out any messages: in order to output messages
#    # on the terminal, just set the standard Gmsh option "General.Terminal" (same
#    # format and meaning as in .geo files):
#    gmsh.option.setNumber("General.Terminal", 1)
#    
#    # Next we add a new model named "t1" (if gmsh.model.add() is not called a new
#    # unnamed model will be created on the fly, if necessary):
#    gmsh.model.add("circle")
#      
#    gmsh.model.geo.addPoint(x2,y2,0,lc,1)
#    gmsh.model.geo.addPoint(x1,y2,0,lc,2)
#    gmsh.model.geo.addPoint(x1,y1,0,lc,3)
#    gmsh.model.geo.addPoint(x2,y1,0,lc,4)
#      
#    gmsh.model.geo.addPoint((x1+x2)/2, (y1+y2)/2, 0,lc2, 6)
#    gmsh.model.geo.addPoint((x1+x2)/2 + R, (y1+y2)/2 , 0, lc2, 5)    
#    gmsh.model.geo.addPoint((x1+x2)/2, (y1+y2)/2 + R , 0, lc2, 7)
#    gmsh.model.geo.addPoint((x1+x2)/2 - R, (y1+y2)/2, 0, lc2, 9)    
#    gmsh.model.geo.addPoint((x1+x2)/2, (y1+y2)/2 - R, 0, lc2, 10)
#    
#    gmsh.model.geo.addLine(1, 2, 1)
#    gmsh.model.geo.addLine(2, 3, 2)
#    gmsh.model.geo.addLine(3, 4, 3)
#    gmsh.model.geo.addLine(4, 1, 4)
#    
#    gmsh.model.geo.addCircleArc(5, 6, 7, 5)
#    gmsh.model.geo.addCircleArc(7, 6, 9, 6)
#    gmsh.model.geo.addCircleArc(9, 6, 10, 7)
#    gmsh.model.geo.addCircleArc(10, 6, 5, 8)
#    
#    gmsh.model.geo.addCurveLoop([1,2,3,4],1)
#    gmsh.model.geo.addCurveLoop([5,6,7,8],2)
#    
#    gmsh.model.geo.addPlaneSurface([2,1],11)
#    
#    gmsh.model.addPhysicalGroup(2, [11], 11)
#    gmsh.model.setPhysicalName(2, 11, "DOMAINE")
# 
#    gmsh.model.geo.synchronize()
#    gmsh.model.mesh.generate()
#     
#    nodeTags, coord, parametricCoord = gmsh.model.mesh.getNodes()
#    
#    # Reshape to get right format
#    points = np.reshape(coord, (int(coord.size / 3), 3))
#    
#    element_types, element_tags, node_tags = gmsh.model.mesh.getElements()
#    
#    Bord, Connectivity, Coin = node_tags
#    CT = Connectivity.reshape(int(Connectivity.size/3),3)-1
#    
#    node=[]
#    nodeTaglist = nodeTags.tolist()
#    nodenumbersread = []
#    for icells in range(0,len(CT)):  
#        for i in range(0,3):  
#            if (CT[icells][i] in nodenumbersread) == 0:
#                nodenumbersread.append(CT[icells][i]) 
#                point_index = nodeTaglist.index(CT[icells][i] + 1)
#                node = np.append(node, points[point_index,0])
#                node = np.append(node, points[point_index,1] )
#                node = np.append(node, 0)
#    
#    node = node.reshape(int(node.size/3),3)
#    
#    nodesreordered =  np.zeros((len(node), 3))
#    for i in range(0,node.shape[0]):
#        nodesreordered[nodenumbersread[i]] = node[i]
#
#    # End Generate mesh
#    gmsh.finalize()
#        
#    xx=nodesreordered[:,0]
#    yy=nodesreordered[:,1]
#    
#    aretei,aretef = TriAretes(xx,yy,CT)
#    
##identifie les 4 frontières Ouest:0, Sud:1, Est:2, Nord:3    
#    for iare in range(0,aretef.shape[0]):
#        xmid = 0.5*(xx[aretef[iare,0]]+xx[aretef[iare,1]])
#        ymid = 0.5*(yy[aretef[iare,0]]+yy[aretef[iare,1]])
#        if(xmid < x1+0.1*lc):
#            aretef[iare,4]=0
#        elif(ymid < y1+0.1*lc):
#            aretef[iare,4]=1
#        elif(xmid > x2-0.1*lc):
#            aretef[iare,4]=2
#        elif(ymid > y2-0.1*lc):
#            aretef[iare,4]=3
#        else:    
#            aretef[iare,4]=4
#            
#    aretes = np.append(aretef,aretei,axis=0)
# 
#    return xx,yy, CT, aretes

"""
FONCTION :  meshiowritevtk(xx,yy,tri,NodeDataDict,TriDataDict)

Fonction qui crée un fichier de solution en format *.vtk pour Paraview 
à partir d'une triangulation. Les champs de solution peuvent être aux noeuds 
ou aux triangles.

INPUT: 
xx, yy        : array de coordonnées des noeuds du maillage
tri           : connectivité des triangles 
NodeDataDict  : dictionnaire des solutions au niveau des noeuds. 
                Exemple:   NodeDataDict = {'x':xx,'y':yy}
                Les données peuvent être des scalaires ou 
                des vecteurs de dimension (nnod,3)
TriDataDict   : dictionnaire des solution au niveau des triangles 
                Exemple:   TriDataDict  = {'Solxt':Solxt,'Area':Area}
filename      : le nom du fichier qui sera écrit (sans extension, sera .vtk)                
OUTPUT: 
Le fichier écrit sur disque est filename.vtk

"""
import numpy as np
import meshio

def meshiowritevtk(xx,yy,tri,NodeDataDict,TriDataDict,filename):
    """Ecriture sur maillage triangles d'un format vtk pour Paraview"""
    node3 = np.ndarray((xx.size,3))
    node3[:,0]=xx
    node3[:,1]=yy
    node3[:,2] = np.zeros(xx.size)
    points = node3
    cells = { "triangle": tri }
    fileExt = filename + '.vtk'
    meshio.write_points_cells(
    fileExt,
    points,
    cells, 
    file_format='vtk-binary',
    point_data = NodeDataDict,
    cell_data={"triangle":TriDataDict}
    )

    return None

