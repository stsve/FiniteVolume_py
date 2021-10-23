import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import mec6616
import numpy as np
import time
import sympy as sp
import scipy.sparse as sprs

#%%
def varInit(divX,divY):
    global xx,yy,tri,are,cenTri,MTri1,fig1,ax1,bcdata,k,n_node,centri,xmin,xmax,ymin,ymax,fT
    
    divx = divX                          # Nombre de divisions en x
    divy = divY                          # Nombre de divisions en y
    n_node = (divx+1)*(divy+1);          # Nombre total de noeuds
    
    xmin = -1;                          # Limite inférieure du domaine en x
    xmax = 1;                           # Limite supérieure du domaine en x
    ymin = -1;                          # Limite inférieure du domaine en y
    ymax = 1;                           # Limite supérieure du domaine en y
    
    k = 0.01
    
    x,y = sp.symbols('x y')
    T = (1-x**2)*(4-y**2)
    fT = sp.lambdify([x,y], T,'numpy')
    
    
#    xx,yy,tri,are = mec6616.RectMesh(xmin,xmax,ymin,ymax,divx,divy)
    xx,yy,tri,are = mec6616.RectGmsh(xmin,xmax,ymin,ymax,(xmax-xmin)/divx)
    MTri1 = mtri.Triangulation(xx, yy,tri)
    
    if triangulation:   
        fig1, ax1 = plt.subplots()
        ax1.set_aspect('equal')
        ax1.triplot(MTri1, 'k.-', lw=1)
        ax1.set_title('Triangulation Delaunay du domaine')
        ax1.set_adjustable("datalim")
        ax1.set_xlim(xmin - 0.1*(xmax-xmin), xmax + 0.1*(xmax-xmin))
#       ax2.set_ylim(1e-1, 1e3)
    
#       annoter les noeuds
        for i in range (0,xx.size):
            plt.annotate(str(i),(xx[i],yy[i]))
        #   annoter les triangles
        for i in range (0,tri.shape[0]):
            xc = (xx[tri[i,0]]+xx[tri[i,1]]+xx[tri[i,2]])/3.0
            yc = (yy[tri[i,0]]+yy[tri[i,1]]+yy[tri[i,2]])/3.0    
            plt.annotate(str(i),(xc,yc)) 
            
        plt.show()
    
    centri = np.zeros((np.size(tri,0),2))
    
    u = 0;
    for t in tri:
        centri[u,0] = np.sum(xx[t])/3
        centri[u,1] = np.sum(yy[t])/3
        u += 1
    
    #--------------------------------------------------------#
    #                   Conditions limites                   #
    #--------------------------------------------------------#
    
    bcdata = (['Dirichlet',fT],['Dirichlet',fT],['Dirichlet',fT],['Dirichlet',fT])

# %%
def normGen():
    global nx,ny,mAx,mAy,dA,dxA,dyA,N
    dxA = xx[are[:,1]]-xx[are[:,0]]
    dyA = yy[are[:,1]]-yy[are[:,0]]
    mAx = 0.5*(xx[are[:,0]]+xx[are[:,1]]) # Coordonnées x du milieu sur chaque arete
    mAy = 0.5*(yy[are[:,0]]+yy[are[:,1]]) # Coordonnées y du milieu sur chaque arete
    
    dA = np.sqrt(np.square(dxA)+np.square(dyA))
    
    nx = dyA/dA;
    ny = -dxA/dA;
    
    N = np.zeros((np.size(nx),2))
    N[:,0] = nx
    N[:,1] = ny


# %%
def ksiGen():
    
    ti = np.where(are[:,3] != -1)
    ksi = np.zeros((np.size(ti),4))
    
    xksi = centri[are[ti[0][0]:ti[0][-1]+1,3],0] - centri[are[ti[0][0]:ti[0][-1]+1,2],0]
    yksi = centri[are[ti[0][0]:ti[0][-1]+1,3],1] - centri[are[ti[0][0]:ti[0][-1]+1,2],1]
    
    
    dksi = np.sqrt(np.square(xksi)+np.square(yksi))
    
    xksiBound = mAx[0:ti[0][0]] - centri[are[0:ti[0][0],2],0]
    yksiBound = mAy[0:ti[0][0]] - centri[are[0:ti[0][0],2],1]
    
    
    # Calcul de Delta Ksi réel
    # ------------------------
    
    Xksi = np.hstack((xksiBound,xksi))
    Yksi = np.hstack((yksiBound,yksi))
    
    DKSI = np.sqrt(np.square(Xksi)+np.square(Yksi))
    
    # -----------------------
    dksiBound = np.sqrt(np.square(xksiBound)+np.square(yksiBound))
    
    xksiBound = xksiBound/dksiBound
    yksiBound = yksiBound/dksiBound
    
    
    xksi = xksi/dksi
    yksi = yksi/dksi
    
    ksi[:,0] = xksi
    ksi[:,1] = yksi
    ksi[:,2] = are[ti[0][0]:ti[0][-1]+1,2]
    ksi[:,3] = are[ti[0][0]:ti[0][-1]+1,3]
    
    Xksi = np.hstack((xksiBound,xksi))
    Yksi = np.hstack((yksiBound,yksi))
    
    KSI = np.zeros((np.size(Xksi),2))
    KSI[:,0] = Xksi
    KSI[:,1] = Yksi
    
    return KSI, DKSI

#%%
    
def nodal_interpolation(Sol_aretes):

    A = np.zeros((np.size(xx),np.shape(tri)[0]))
    
    #construction matrice rescensant tous les triangles qui vont impacter un noeud
    for arete in are:
        A[arete[0]][arete[2]]=1
        A[arete[1]][arete[2]]=1
        if (arete[3] != -1):
            A[arete[0]][arete[3]]=1
            A[arete[1]][arete[3]]=1
    
    #ajuste l'impact de chacun de ces triangles en fonction du nbr qu'ils sont à influer chaque noeud
    for a in A:
        a/=sum(a[:])  
    
    #résolution avec la valeur de T du centre des triangles
        
    As = sprs.csr_matrix(A)
#    Sol = np.matmul(A,Sol_aretes)
    Sol = As.dot(Sol_aretes)
    
    #modifie les noeuds limites qui ont des conditions Dirichlet
    u=0;
    for arete in are:
        if (arete[3] == -1 and (bcdata[arete[4]][0] =='Dirichlet')):
            Sol[arete[0]]=bcdata[arete[4]][1](mAx[u],mAy[u])
            Sol[arete[1]]=bcdata[arete[4]][1](mAx[u],mAy[u])
        u+=1
    
    return (Sol)

# %%
def airTri(triangles):
    
    rTri = np.zeros((np.size(triangles,0)))
    
    u=0;
    for tri in triangles:
        a = np.sqrt(np.square(xx[tri[1]]-xx[tri[0]])+  np.square(yy[tri[1]]-yy[tri[0]]))
        b = np.sqrt(np.square(xx[tri[2]]-xx[tri[0]])+  np.square(yy[tri[2]]-yy[tri[0]]))
        c = np.sqrt(np.square(xx[tri[2]]-xx[tri[1]])+  np.square(yy[tri[2]]-yy[tri[1]]))
        
        p = (a+b+c)/2
        
        rTri[u] = np.sqrt(p*(p-a)*(p-b)*(p-c))
        u+=1
        
    return rTri


#%%

def etaGen():
    ETA = np.zeros((np.size(dxA,0),2))
    etax = dxA/dA
    etay = dyA/dA
    deta = dA
    
    ETA[:,0] = etax
    ETA[:,1] = etay
    
    return ETA,deta

#%%
def solveur2D(numelx,numely):
    global PNKSI, PNN,A,ETA,KSI,deta,vel,X,Y,solPhi,tim2,flag,solPhip
    
    varInit(numelx,numely)
    normGen()
    KSI,DKSI = ksiGen()
    ETA,deta = etaGen()
    aireTri = airTri(tri)
    
    print('\nComputing for '+str(numelx)+'x'+str(numely)+' mesh')
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    
    X = centri[:,0]
    Y = centri[:,1]
    
    Source = -k*(2*X**2 + 2*Y**2 - 10) + X*Y*(-6.0*X**4*Y**2 + 6.0*X**4 + 12.0*X**2*Y**2 -12.0*X**2 - 6.0*Y**2 + 6.0)

    q = Source
    rho = 1
    Cp = 1
    
    x,y = sp.symbols('x y')
    U = (2*(x**2)-(x**4)-1)*(y - y**3)
    V = -(2*(y**2)-(y**4)-1)*(x - x**3)
    
    uu = sp.lambdify([x,y], U,'numpy')
    vv = sp.lambdify([x,y], V,'numpy')
    
    vel = np.zeros((np.size(tri,0),2))
    for i in range(0,np.size(centri,0)):
        vel[i,0] = uu(centri[i,0],centri[i,1])
        vel[i,1] = vv(centri[i,0],centri[i,1])
    
    
    if triangulation:
        ax1.quiver(mAx,mAy,nx,ny,width=0.006)
        ax1.scatter(centri[:,0],centri[:,1],marker = '.',c='b')
        ax1.quiver(mAx,mAy,KSI[:,0],KSI[:,1],width=0.003,color='g')
        ax1.quiver(mAx,mAy,ETA[:,0],ETA[:,1],width=0.004,color='r')
        plt.show()
        plt.show()
    
    A = np.zeros((np.size(tri,0),np.size(tri,0)))
    
    if sum(q) == 0: 
        b = np.zeros((np.size(tri,0)))
    else:
        b = np.ones((np.size(tri,0)))*aireTri*q
    
    PNKSI = np.zeros((np.size(are,0)))
    PNN = np.zeros((np.size(are,0)))
    D = np.zeros((np.size(are,0)))
    F = np.zeros((np.size(are,0)))
    
    tim = time.time()
    
    u = 0;
    for i in are:
        ndot = np.dot(N[u,:],N[u,:])
        PNN[u] = ndot
        PNKSI[u] = np.dot(N[u,:],KSI[u,:])
        lmbda = ((k*deta[u])/(DKSI[u]))*(ndot/PNKSI[u])
        D[u] = lmbda
        F[u] = rho*Cp*np.dot(N[u,:],np.array([uu(mAx[u],mAy[u]),vv(mAx[u],mAy[u])]))*deta[u]
        
        if (i[3]== -1 and bcdata[i[4]][0] != 'Neumann'):
            A[i[2],i[2]] += D[u] + max(F[u],0)
            
        if (i[3]== -1 and bcdata[i[4]][0] == 'Neumann'):
             A[i[2],i[2]] += F[u]
                  
        if (i[3] != -1 and not upwind):
            A[i[2],i[3]] += -D[u] + F[u]/2
            A[i[3],i[2]] += -D[u] - F[u]/2
            A[i[3],i[3]] += D[u] - F[u]/2
            A[i[2],i[2]] += D[u] + F[u]/2
            
        if (i[3] != -1 and upwind):
            A[i[2],i[3]] += -D[u] - max(0,-F[u])
            A[i[3],i[2]] += -D[u] - max(F[u],0)
            A[i[3],i[3]] += D[u] + max(0,-F[u])
            A[i[2],i[2]] += D[u] + max(F[u],0)
            
        u +=1
             
    u = 0;  
    for i in are:
        if (i[3] == -1 and bcdata[i[4]][0]=='Dirichlet'):
            b[i[2]] += D[u]*bcdata[i[4]][1](mAx[u],mAy[u]) + max(0,-F[u])*bcdata[i[4]][1](mAx[u],mAy[u])
            
        u += 1
    
    tim = time.time() - tim
#    print('pre-construction time: ',tim)
    
    solNode = np.zeros((np.size(xx)))

    flag = True
    it = 0
    solPhip = np.zeros((np.size(tri,0)))
    solPhi = np.zeros((np.size(tri,0)))
    

    from tqdm import tqdm
    pbar = tqdm(total=15)
    
    tim2 = time.time()
    while flag and it < 150:
        
        pbar.set_description('Processing iteration '+ str(it ))
        
        S = np.zeros((np.size(b)))
        
        u = 0;
        for i in are:
            
            pksieta = np.dot(KSI[u,:],ETA[u,:])
            Scd = -(((k*dA[u])/deta[u]) * (pksieta/PNKSI[u]))*(solNode[i[1]]-solNode[i[0]])
              
            if i[3] == -1:
                S[i[2]] += Scd
            else:
                S[i[2]] += Scd
                S[i[3]] += -Scd 
            u += 1
        
        B = b + S;    
        
        solPhip = np.copy(solPhi);
        solPhi = np.linalg.solve(A,B)
        
        solNode = nodal_interpolation(solPhi)
        
        if max(np.abs(solPhi - solPhip)) < 1e-8:
            flag = False
#        print('iteration:',it)
#        print('-----------')    
        
        pbar.update(1)
        it+= 1
        
    pbar.close()
    
    tim2 = time.time()-tim2 
    
    return A,b



#%%
#-----------------------------------------------------------------------------#          
#                               ***  MAIN   ***
# -----------------------------------------------------------------------------    

#-----------------------------------------------------------------------------#
#                   Méthode des solutions Manufacturées                       #
#-----------------------------------------------------------------------------#

triangulation = False
upwind =  not True
A,b = solveur2D(20,20)

if triangulation:
    ax1.quiver(mAx,mAy,nx,ny,width=0.006)
    ax1.scatter(centri[:,0],centri[:,1],marker = '.',c='b')
    ax1.quiver(mAx,mAy,KSI[:,0],KSI[:,1],width=0.003,color='g')
    ax1.quiver(mAx,mAy,ETA[:,0],ETA[:,1],width=0.004,color='r')
    plt.show()

print('__________________________________________________________________') 
print('\nConverged solution ')
print('------------------------------------------------------------------')
print('total processing time:',tim2,'s')
print('------------------------------------------------------------------')


fig2, ax2 = plt.subplots()
tcf = ax2.tripcolor(MTri1, facecolors=solPhi,edgecolor = 'k',cmap = 'viridis')
fig2.colorbar(tcf)
ax2.quiver(centri[:,0],centri[:,1],vel[:,0],vel[:,1],width=0.003)
ax2.set_title('Contour plot on triangulation \n Upwind: '+str(upwind))
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y Axis')
ax2.axes.set_aspect('equal')
plt.show()

SolAna = fT(X,Y)

fig3, ax3 = plt.subplots()
tcf = ax3.tripcolor(MTri1, facecolors=SolAna, edgecolors='k',cmap = 'viridis')
fig3.colorbar(tcf)
ax3.set_title('Solution analytique')
ax3.set_xlabel('X axis')
ax3.set_ylabel('Y Axis')
ax3.axes.set_aspect('equal')
plt.show()

#   Erreur aux triangles MMS

errTri = np.abs(solPhi-SolAna)
normL2 = np.sqrt((1/np.size(solPhi))*np.sum(np.square(errTri)));

print('Erreur en norme L2:',normL2)
print('------------------------------------------------------------------')

fig4, ax4 = plt.subplots()
tcf = ax4.tripcolor(MTri1, facecolors=errTri, edgecolors='k',cmap = 'viridis')
fig4.colorbar(tcf)
ax4.set_title('Erreur aux triangles (MMS) \n Upwind: '+str(upwind))
ax4.set_xlabel('X axis')
ax4.set_ylabel('Y Axis')
ax4.axes.set_aspect('equal')
plt.show()

#%% Orde de convergence

#nraff = np.array([2,4,16,32])
#
#NORME = np.zeros((np.size(nraff)))
#ordre = np.zeros((np.size(nraff))-1)
#
#for ii in range(0,np.size(nraff)):
#    
#    nelm = nraff[ii]
#    
#    solveur2D(nelm,nelm)
#    
#    SolAna = fT(X,Y)
#    
#    errTri = np.abs(solPhi-SolAna)
#    normL2 = np.sqrt((1/np.size(solPhi))*np.sum(np.square(errTri)));
#    
#    NORME[ii]=normL2
#    
#
#for n in range(1,np.size(nraff)):
#    ordre[n-1] = np.log(NORME[n-1]/NORME[n])/np.log(2)
#    
#L = xmax-xmin    
#
#fig=plt.figure()
#plt.loglog(L/nraff,NORME,'-ko')
#plt.title('Ordre de convergence \n Upwind:'+str(upwind)+' | Pente = ' + str(round(np.average(ordre),2)))
#plt.ylabel('$Erreur$')
#plt.xlabel('$\Delta h$')
#ax = fig.add_subplot(111)
#ax.grid(b=True, which='minor', color='grey', linestyle='--')
#ax.grid(b=True, which='major', color='k', linestyle='-')
#    
#print(NORME)
#print(ordre)













