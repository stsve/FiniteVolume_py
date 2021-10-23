import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import mec6616
import numpy as np
import time
import sympy as sp
import scipy.sparse as sprs

#%%
#
#x,y,u,v = sp.symbols('x y u v')
#u = (2*x**2 - x**4 - 1)*(y-y**3)
#v = -(2*y**2 - y**4 - 1)*(x-x**3)
#T = (1-x**2)*(4-y**2)
#fT = sp.lambdify([x,y], T,'numpy')
#mu = 0.01
#
#convx = sp.simplify(sp.diff(u*u,x)+sp.diff(v*u,y))
#diffx = sp.simplify(sp.diff(sp.diff(u,x),x)+sp.diff(sp.diff(u,y),y))
#Sourcex = convx - mu*diffx
#
#convy = sp.simplify(sp.diff(u*v,x)+sp.diff(v*v,y))
#diffy = sp.simplify(sp.diff(sp.diff(v,x),x)+sp.diff(sp.diff(v,y),y))
#Sourcey = convy - mu*diffy
#
#velTot = sp.simplify((u**2 + v**2)**(1/2))
#
#fu = sp.lambdify([x,y],u,'numpy')
#fv = sp.lambdify([x,y],v,'numpy')
#fSx = sp.lambdify([x,y],Sourcex,'numpy')
#fSy = sp.lambdify([x,y],Sourcey,'numpy')
#fVel = sp.lambdify([x,y],velTot,'numpy')
#
#
##print('convx -> ',convx)
##print('diffx -> ',diffx)
#print('Sourcex -> ',Sourcex)
#
#print('-----------------------------------------------------------------------------------------')
#
##print('convy -> ',convy)
##print('diffy -> ',diffy)
#print('Sourcey -> ',Sourcey)
#


#%%
def varInit(divX,divY):
    global xx,yy,tri,are,cenTri,MTri1,fig1,ax1,bcdata_u,bcdata_v,k,n_node,centri,xmin,xmax,ymin,ymax,fu,fv,fVel
    
    divx = divX                          # Nombre de divisions en x
    divy = divY                          # Nombre de divisions en y
    n_node = (divx+1)*(divy+1);          # Nombre total de noeuds
    
    xmin = 0;                          # Limite inférieure du domaine en x
    xmax = 1;                       # Limite supérieure du domaine en x
    ymin = 0;                          # Limite inférieure du domaine en y
    ymax = 1;                       # Limite supérieure du domaine en y
    
    x,y,u,v = sp.symbols('x y u v')
    u = (2*x**2 - x**4 - 1)*(y-y**3)
    v = -(2*y**2 - y**4 - 1)*(x-x**3)
    velTot = sp.simplify((u**2 + v**2)**(1/2))
    fVel = sp.lambdify([x,y],velTot,'numpy')
    
    fu = sp.lambdify([x,y],u,'numpy')
    fv = sp.lambdify([x,y],v,'numpy')
    
    
    xx,yy,tri,are = mec6616.RectMesh(xmin,xmax,ymin,ymax,divx,divy)
#    xx,yy,tri,are = mec6616.RectGmsh(xmin,xmax,ymin,ymax,(xmax-xmin)/divx)
    MTri1 = mtri.Triangulation(xx, yy,tri)
    
    if pltTriang:
        fig1, ax1 = plt.subplots()
        ax1.set_aspect('equal')
        ax1.triplot(MTri1, 'k.-', lw=1)
        ax1.set_title('Triangulation Delaunay du domaine')
        ax1.set_adjustable("datalim")
        ax1.set_xlim(xmin - 0.1*(xmax-xmin), xmax + 0.1*(xmax-xmin))
    #    ax2.set_ylim(1e-1, 1e3)
        
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
    
    bcdata_u = (['Dirichlet',fu],['Dirichlet',fu],['Dirichlet',fu],['Dirichlet',fu])
    bcdata_v = (['Dirichlet',fv],['Dirichlet',fv],['Dirichlet',fv],['Dirichlet',fv])
    
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
    
#%%  
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
        if (arete[3] == -1 and (bcdata_u[arete[4]][0] =='Dirichlet')):
            Sol[arete[0]]=bcdata_u[arete[4]][1](mAx[u],mAy[u])
            Sol[arete[1]]=bcdata_u[arete[4]][1](mAx[u],mAy[u])
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
def msolv2D(numelx,numely):
    global PNKSI, PNN,A,ETA,KSI,deta,vel,X,Y,solU,solV,tim2,flag,solPhip,fSx,fSy,F,AA
    
    varInit(numelx,numely)
    normGen()
    KSI,DKSI = ksiGen()
    ETA,deta = etaGen()
    aireTri = airTri(tri)
    
    x,y,uu,vv = sp.symbols('x y uu vv')
    uu = (2*x**2 - x**4 - 1)*(y-y**3)
    vv = -(2*y**2 - y**4 - 1)*(x-x**3)
#    T = (1-x**2)*(4-y**2)
#    fT = sp.lambdify([x,y], T,'numpy')
    mu = 0.01
    rho = 1
    
    convx = sp.simplify(sp.diff(uu*uu,x)+sp.diff(vv*uu,y))
    diffx = sp.simplify(sp.diff(sp.diff(uu,x),x)+sp.diff(sp.diff(uu,y),y))
    Sourcex = convx - mu*diffx
    
    convy = sp.simplify(sp.diff(uu*vv,x)+sp.diff(vv*vv,y))
    diffy = sp.simplify(sp.diff(sp.diff(vv,x),x)+sp.diff(sp.diff(vv,y),y))
    Sourcey = convy - mu*diffy
    
#    velTot = sp.simplify((uu**2 + vv**2)**(1/2))
    
    fu = sp.lambdify([x,y],uu,'numpy')
    fv = sp.lambdify([x,y],vv,'numpy')
    fSx = sp.lambdify([x,y],Sourcex,'numpy')
    fSy = sp.lambdify([x,y],Sourcey,'numpy')
#    fVel = sp.lambdify([x,y],velTot,'numpy')
    
    X = centri[:,0]
    Y = centri[:,1]
    
    Sx = fSx(X,Y)
    Sy = fSy(X,Y)
    
#    Src = np.sqrt(np.square(Sx)+np.square(Sy))

    qx = Sx
    qy = Sy
#    rho = 1
#    Cp = 1
      
    if pltTriang:
        ax1.quiver(mAx,mAy,nx,ny,width=0.006)
        ax1.scatter(centri[:,0],centri[:,1],marker = '.',c='b')
        ax1.quiver(mAx,mAy,KSI[:,0],KSI[:,1],width=0.003,color='g')
        ax1.quiver(mAx,mAy,ETA[:,0],ETA[:,1],width=0.004,color='r')
        plt.show()
        plt.show()
    
    A = np.zeros((np.size(tri,0),np.size(tri,0)))
    AA = np.zeros((np.size(tri,0),np.size(tri,0)))
    
#    if sum(qx) == 0 and sum(qy)==0: 
#        bx = np.zeros((np.size(tri,0)))
#        by = np.zeros((np.size(tri,0)))
#    else:
#        bx = np.ones((np.size(tri,0)))*aireTri*qx
#        by = np.ones((np.size(tri,0)))*aireTri*qy
    
    PNKSI = np.zeros((np.size(are,0)))
    PNN = np.zeros((np.size(are,0)))
    D = np.zeros((np.size(are,0)))
    
    F = np.zeros((np.size(are,0)))
    
    up = np.zeros((np.size(tri,0)))
    vp = np.zeros((np.size(tri,0)))
    
#    upp= np.zeros((np.size(tri,0)))
#    vpp = np.zeros((np.size(tri,0)))
    
    itt = 0
    while itt < 40:
         
        if sum(qx) == 0 and sum(qy)==0: 
            bx = np.zeros((np.size(tri,0)))
            by = np.zeros((np.size(tri,0)))
        else:
            bx = np.ones((np.size(tri,0)))*aireTri*qx
            by = np.ones((np.size(tri,0)))*aireTri*qy
        
        Fx = rho*up
        Fy = rho*vp
        
    #    tim = time.time()
        
#        print('Avant:',A)
#        A = np.copy(AA)
#        print('Apres:',A)
        
        A = np.zeros((np.size(tri,0),np.size(tri,0)))
        u = 0;
        for i in are:
            ndot = np.dot(N[u,:],N[u,:])
            PNN[u] = ndot
            PNKSI[u] = np.dot(N[u,:],KSI[u,:])
            lmbda = ((mu*deta[u])/(DKSI[u]))*(ndot/PNKSI[u])
            D[u] = lmbda
    #        F[u] = rho*np.dot(N[u,:],np.array([fu(mAx[u],mAy[u]),fv(mAx[u],mAy[u])]))*deta[u]
            if i[3] != -1:
                F[u] = np.dot(N[u,:],np.array([(1/2*(Fx[i[2]]+Fx[i[3]])),(1/2*(Fy[i[2]]+Fy[i[3]]))]))*deta[u]
            else:
#                F[u] = np.dot(N[u,:],np.array([(1/2*(Fx[i[2]]+bcdata_u[i[4]][1](mAx[u],mAy[u]))),(1/2*(Fy[i[2]]+bcdata_v[i[4]][1](mAx[u],mAy[u])))]))*deta[u]
                 F[u] = np.dot(N[u,:],np.array([(rho*bcdata_u[i[4]][1](mAx[u],mAy[u])),(rho*bcdata_v[i[4]][1](mAx[u],mAy[u]))]))*deta[u]
                 
            if (i[3]== -1 and bcdata_u[i[4]][0] != 'Neumann'):
                A[i[2],i[2]] += D[u] + max(F[u],0)
                
            if (i[3]== -1 and bcdata_u[i[4]][0] == 'Neumann'):
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
        
        if sum(F)==0:
            AA = np.copy(A)
                 
        u = 0;  
        for i in are:
            if (i[3] == -1 and bcdata_u[i[4]][0]=='Dirichlet'):
                bx[i[2]] += D[u]*bcdata_u[i[4]][1](mAx[u],mAy[u]) + max(0,-F[u])*bcdata_u[i[4]][1](mAx[u],mAy[u])
                
            if (i[3] == -1 and bcdata_v[i[4]][0]=='Dirichlet'):
                 by[i[2]] += D[u]*bcdata_v[i[4]][1](mAx[u],mAy[u]) + max(0,-F[u])*bcdata_v[i[4]][1](mAx[u],mAy[u])    
            u += 1
        
#        if sum(F)==0:
#            bbx = np.copy(bx)
#            bby = np.copy(by)
        
        
    #    tim = time.time() - tim
    #    print('pre-construction time: ',tim)
        
        solNodex = np.zeros((np.size(xx)))
        solNodey = np.zeros((np.size(yy)))
    
        flag = True
        it = 0
        solPhipx = np.zeros((np.size(tri,0)))
        solPhipy = np.zeros((np.size(tri,0)))
        solU = np.zeros((np.size(tri,0)))
        solV = np.zeros((np.size(tri,0)))
        
    #    tim2 = time.time()
        while flag and it < 150:
    #        print('*** CHECK 1 ***')
            Sxx = np.zeros((np.size(bx)))
            Syy = np.zeros((np.size(by)))
            
            u = 0;
            for i in are:
                pksieta = np.dot(KSI[u,:],ETA[u,:])
                Scdx = -(((mu*dA[u])/deta[u]) * (pksieta/PNKSI[u]))*(solNodex[i[1]]-solNodex[i[0]])
                Scdy = -(((mu*dA[u])/deta[u]) * (pksieta/PNKSI[u]))*(solNodey[i[1]]-solNodey[i[0]])
                  
                if i[3] == -1:
                    Sxx[i[2]] += Scdx
                    Syy[i[2]] += Scdy
                else:
                    Sxx[i[2]] += Scdx
                    Sxx[i[3]] += -Scdx
                    Syy[i[2]] += Scdy
                    Syy[i[3]] += -Scdy
                u += 1
            
            B1 = bx + Sxx;    
            B2 = by + Syy;
    
            solPhipx = np.copy(solU);
            solPhipy = np.copy(solV);
            
            solU = np.linalg.solve(A,B1)
            solV = np.linalg.solve(A,B2)
        #    solNode = intNode(are,tri,xx,yy,solPhi,bcdata,n_node,xmin,xmax,ymin,ymax)
        
            solNodex = nodal_interpolation(solU)
            solNodey = nodal_interpolation(solV)
            
            if max(np.abs(solU - solPhipx)) < 1e-8 and max(np.abs(solV - solPhipy)) < 1e-8:
                flag = False
    
            print('iteration:',it)
            print('-----------')
            it+= 1
        
    #    alpha= 0.6
    #    
    #    
    #    up = alpha*solU + (1-alpha)*upp
    #    vp = alpha*solV + (1-alpha)*vpp

        up = np.copy(solU)
        vp = np.copy(solV)
        
        itt += 1
    
    
#    tim2 = time.time()-tim2 
    
    return A,bx,by

#%%
    
pltTriang = not True
upwind = not True

#varInit(20,20)
msolv2D(20,20)

X = centri[:,0]
Y = centri[:,1]

Sx = fSx(X,Y)
Sy = fSy(X,Y)

U = fu(X,Y)
V = fv(X,Y)

VEL = np.sqrt(np.square(U)+np.square(V))

S = np.sqrt(np.square(Sx)+np.square(Sy))

fig1, ax1 = plt.subplots()
tcf = ax1.tripcolor(MTri1, facecolors = VEL,edgecolor = 'k',cmap = 'viridis')
fig1.colorbar(tcf)
ax1.quiver(centri[:,0],centri[:,1],U,V,width=0.003)
ax1.set_title('Champs de vitesse donné \n (Analytique)')
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y Axis')
ax1.axes.set_aspect('equal')
plt.show()

solVelocity = np.sqrt(np.square(solU)+np.square(solV))

fig2, ax2 = plt.subplots()
tcf = ax2.tripcolor(MTri1, facecolors=solVelocity,edgecolor = 'k',cmap = 'viridis')
fig2.colorbar(tcf)
ax2.quiver(centri[:,0],centri[:,1],solU,solV,width=0.003)
ax2.set_title('Solution numérique \n sur triangulation')
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y Axis')
ax2.axes.set_aspect('equal')
plt.show()


#%% Coupes  en X

Xu = np.unique(X)
Yu = np.unique(Y)

a = int(np.floor(2*np.size(Xu)/3))

pos = np.where(X==Xu[0])
vcoupe = np.zeros((np.size(pos)))
YY = np.zeros((np.size(pos)))
u = 0
for ii in range(0,np.size(pos)):
    vcoupe[ii] = solVelocity[pos[0][ii]]
    YY[ii] = Y[pos[0][ii]]
    u+=1 
        
fig3, ax3 = plt.subplots()
ax3.scatter(YY,vcoupe, marker = 'o',c = 'k',label='Solution numérique aux triangles')
ax3.plot(np.sort(Y[pos[0]]),fVel(X[pos[0]],np.sort(Y[pos[0]])),'-r',label='Solution analytique')
ax3.set_title('Comparaison solution numérique | Analytique \n Coupe x='+str(round(Xu[0],3)))
ax3.set_xlabel('y')
ax3.set_ylabel('Vitesse')
ax3.axes.set_aspect('equal')
plt.legend()
plt.show()

pos = np.where(X==Xu[a])
vcoupe = np.zeros((np.size(pos)))
YY = np.zeros((np.size(pos)))
u = 0
for ii in range(0,np.size(pos)):
    vcoupe[ii] = solVelocity[pos[0][ii]]
    YY[ii] = Y[pos[0][ii]]
    u+=1 
        
fig4, ax4 = plt.subplots()
ax4.scatter(YY,vcoupe,marker = 'o',c = 'k',label='Solution numérique aux triangles')
ax4.plot(np.sort(Y[pos[0]]),fVel(X[pos[0]],np.sort(Y[pos[0]])),'-r',label='Solution analytique')
ax4.set_title('Comparaison solution numérique | Analytique \n Coupe x='+str(round(Xu[a],3)))
ax4.set_xlabel('y')
ax4.set_ylabel('Vitesse')
ax4.axes.set_aspect('equal')
plt.legend()
plt.show()


pos = np.where(X==Xu[-1])
vcoupe = np.zeros((np.size(pos)))
YY = np.zeros((np.size(pos)))
u = 0
for ii in range(0,np.size(pos)):
    vcoupe[ii] = solVelocity[pos[0][ii]]
    YY[ii] = Y[pos[0][ii]]
    u+=1 
        
fig5, ax5 = plt.subplots()
ax5.scatter(YY,vcoupe,marker = 'o',c = 'k',label='Solution numérique aux triangles')
ax5.plot(np.sort(Y[pos[0]]),fVel(X[pos[0]],np.sort(Y[pos[0]])),'-r',label='Solution analytique')
ax5.set_title('Comparaison solution numérique | Analytique \n Coupe x='+str(round(Xu[-1],3)))
ax5.set_xlabel('y')
ax5.set_ylabel('Vitesse')
plt.legend()
plt.show()


#%% Orde de convergence (Centré)
##
nraff = np.array([2,4,16])
NORME = np.zeros((np.size(nraff)))
ordre = np.zeros((np.size(nraff))-1)

for ii in range(0,np.size(nraff)):
    
    nelm = nraff[ii]
    
    msolv2D(nelm,nelm)
    
    SolAna = fVel(X,Y)
    solVelocity = np.sqrt(np.square(solU)+np.square(solV))
    
    errTri = np.abs(solVelocity-SolAna)
    normL2 = np.sqrt((1/np.size(solVelocity))*np.sum(np.square(errTri)));
    
    NORME[ii]=normL2
    

for n in range(1,np.size(nraff)):
    ordre[n-1] = np.log(NORME[n-1]/NORME[n])/np.log(2)
    
print(NORME)
print(ordre)

L =xmax-xmin

fig=plt.figure()
plt.loglog(L/nraff,NORME,'-ko')
plt.title('Ordre de convergence - LAPP4 \n Upwind:'+str(upwind)+' | Pente = ' + str(round(np.average(ordre),2)))
plt.ylabel('$Erreur$')
plt.xlabel('$\Delta h$')
ax = fig.add_subplot(111)
ax.grid(b=True, which='minor', color='grey', linestyle='--')
ax.grid(b=True, which='major', color='k', linestyle='-')


#%%

upwind = True

#varInit(20,20)
msolv2D(20,20)

X = centri[:,0]
Y = centri[:,1]

Sx = fSx(X,Y)
Sy = fSy(X,Y)

U = fu(X,Y)
V = fv(X,Y)

VEL = np.sqrt(np.square(U)+np.square(V))

S = np.sqrt(np.square(Sx)+np.square(Sy))

fig1, ax1 = plt.subplots()
tcf = ax1.tripcolor(MTri1, facecolors = VEL,edgecolor = 'k',cmap = 'viridis')
fig1.colorbar(tcf)
ax1.quiver(centri[:,0],centri[:,1],U,V,width=0.003)
ax1.set_title('Champs de vitesse donné \n (Analytique)')
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y Axis')
ax1.axes.set_aspect('equal')
plt.show()

solVelocity = np.sqrt(np.square(solU)+np.square(solV))

fig2, ax2 = plt.subplots()
tcf = ax2.tripcolor(MTri1, facecolors=solVelocity,edgecolor = 'k',cmap = 'viridis')
fig2.colorbar(tcf)
ax2.quiver(centri[:,0],centri[:,1],solU,solV,width=0.003)
ax2.set_title('Solution numérique \n sur triangulation')
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y Axis')
ax2.axes.set_aspect('equal')
plt.show()

#%
# Coupes  en X

Xu = np.unique(X)
Yu = np.unique(Y)

a = int(np.floor(2*np.size(Xu)/3))

pos = np.where(X==Xu[0])
vcoupe = np.zeros((np.size(pos)))
YY = np.zeros((np.size(pos)))
u = 0
for ii in range(0,np.size(pos)):
    vcoupe[ii] = solVelocity[pos[0][ii]]
    YY[ii] = Y[pos[0][ii]]
    u+=1 
        
fig3, ax3 = plt.subplots()
ax3.scatter(YY,vcoupe, marker = 'o',c = 'k',label='Solution numérique aux triangles')
ax3.plot(np.sort(Y[pos[0]]),fVel(X[pos[0]],np.sort(Y[pos[0]])),'-r',label='Solution analytique')
ax3.set_title('Comparaison solution numérique | Analytique \n Coupe x='+str(round(Xu[0],3)))
ax3.set_xlabel('y')
ax3.set_ylabel('Vitesse')
ax3.axes.set_aspect('equal')
plt.legend()
plt.show()

pos = np.where(X==Xu[a])
vcoupe = np.zeros((np.size(pos)))
YY = np.zeros((np.size(pos)))
u = 0
for ii in range(0,np.size(pos)):
    vcoupe[ii] = solVelocity[pos[0][ii]]
    YY[ii] = Y[pos[0][ii]]
    u+=1 
        
fig4, ax4 = plt.subplots()
ax4.scatter(YY,vcoupe,marker = 'o',c = 'k',label='Solution numérique aux triangles')
ax4.plot(np.sort(Y[pos[0]]),fVel(X[pos[0]],np.sort(Y[pos[0]])),'-r',label='Solution analytique')
ax4.set_title('Comparaison solution numérique | Analytique \n Coupe x='+str(round(Xu[a],3)))
ax4.set_xlabel('y')
ax4.set_ylabel('Vitesse')
ax4.axes.set_aspect('equal')
plt.legend()
plt.show()


pos = np.where(X==Xu[-1])
vcoupe = np.zeros((np.size(pos)))
YY = np.zeros((np.size(pos)))
u = 0
for ii in range(0,np.size(pos)):
    vcoupe[ii] = solVelocity[pos[0][ii]]
    YY[ii] = Y[pos[0][ii]]
    u+=1 
        
fig5, ax5 = plt.subplots()
ax5.scatter(YY,vcoupe,marker = 'o',c = 'k',label='Solution numérique aux triangles')
ax5.plot(np.sort(Y[pos[0]]),fVel(X[pos[0]],np.sort(Y[pos[0]])),'-r',label='Solution analytique')
ax5.set_title('Comparaison solution numérique | Analytique \n Coupe x='+str(round(Xu[-1],3)))
ax5.set_xlabel('y')
ax5.set_ylabel('Vitesse')
plt.legend()
plt.show()


# Orde de convergence (Centré)

nraff = np.array([2,4,16])
NORME = np.zeros((np.size(nraff)))
ordre = np.zeros((np.size(nraff))-1)

for ii in range(0,np.size(nraff)):
    
    nelm = nraff[ii]
    
    msolv2D(nelm,nelm)
    
    SolAna = fVel(X,Y)
    solVelocity = np.sqrt(np.square(solU)+np.square(solV))
    
    errTri = np.abs(solVelocity-SolAna)
    normL2 = np.sqrt((1/np.size(solVelocity))*np.sum(np.square(errTri)));
    
    NORME[ii]=normL2
    

for n in range(1,np.size(nraff)):
    ordre[n-1] = np.log(NORME[n-1]/NORME[n])/np.log(2)
    
print(NORME)
print(ordre)

L =xmax-xmin

fig=plt.figure()
plt.loglog(L/nraff,NORME,'-ko')
plt.title('Ordre de convergence - LAPP4 \n Upwind:'+str(upwind)+' | Pente = ' + str(round(np.average(ordre),2)))
plt.ylabel('$Erreur$')
plt.xlabel('$\Delta h$')
ax = fig.add_subplot(111)
ax.grid(b=True, which='minor', color='grey', linestyle='--')
ax.grid(b=True, which='major', color='k', linestyle='-')