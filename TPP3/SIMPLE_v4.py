#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 18:10:57 2020

@author: stefanesved
"""

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import mec6616
import numpy as np
import time
import sympy as sp
import scipy.sparse as sprs

#%%
def varInit(divX,divY):
    global xx,yy,tri,are,cenTri,MTri1,fig1,ax1,bcdata_u,bcdata_v,k,n_node,centri,xmin,xmax,ymin,ymax,fu,fv,fVel,bcdata,fP,funPress
    
    divx = divX                          # Nombre de divisions en x
    divy = divY                          # Nombre de divisions en y
    n_node = (divx+1)*(divy+1);          # Nombre total de noeuds
    
    xmin = 0;                          # Limite inférieure du domaine en x
    xmax = 10;                         # Limite supérieure du domaine en x
    ymin = 0;                          # Limite inférieure du domaine en y
    ymax = 10;                         # Limite supérieure du domaine en y
    
    x,y,u,v = sp.symbols('x y u v')
#    u = (2*x**2 - x**4 - 1)*(y-y**3)
    u = 10-x
#    v = -(2*y**2 - y**4 - 1)*(x-x**3)
    v = 0*x+0*y
    velTot = sp.simplify((u**2 + v**2)**(1/2))
    funPress = sp.simplify((x**2+y))
    fVel = sp.lambdify([x,y],velTot,'numpy')
    fP = np.vectorize(sp.lambdify([x,y],funPress,'numpy'))
    
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
    
    bcdata = (['Dirichlet',fP],['Dirichlet',fP],['Dirichlet',fP],['Dirichlet',fP])
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
def nodal_interpolation(Sol_aretes,str):

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
        if str == 'x':
            if (arete[3] == -1 and (bcdata_u[arete[4]][0] =='Dirichlet')):
                Sol[arete[0]]=bcdata_u[arete[4]][1](mAx[u],mAy[u])
                Sol[arete[1]]=bcdata_u[arete[4]][1](mAx[u],mAy[u])
        else:
            if (arete[3] == -1 and (bcdata_v[arete[4]][0] =='Dirichlet')):
                Sol[arete[0]]=bcdata_v[arete[4]][1](mAx[u],mAy[u])
                Sol[arete[1]]=bcdata_v[arete[4]][1](mAx[u],mAy[u])
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
def gradLS(phiIni):
    global ATA,b
    ATA = np.zeros((np.size(tri,0),2,2))
    b = np.zeros((np.size(tri,0),2,1))
    
    u = 0
    for a in are:
        if a[3] != -1:
            ATA[a[2]][0,0] += (centri[a[3],0]-centri[a[2],0])**2
            ATA[a[3]][0,0] += (centri[a[2],0]-centri[a[3],0])**2
            
            ATA[a[2]][0,1] += (centri[a[3],0]-centri[a[2],0])*(centri[a[3],1]-centri[a[2],1])
            ATA[a[3]][0,1] += (centri[a[2],0]-centri[a[3],0])*(centri[a[2],1]-centri[a[3],1])
            
            ATA[a[2]][1,0] += (centri[a[3],0]-centri[a[2],0])*(centri[a[3],1]-centri[a[2],1])
            ATA[a[3]][1,0] += (centri[a[2],0]-centri[a[3],0])*(centri[a[2],1]-centri[a[3],1])
            
            ATA[a[2]][1,1] += (centri[a[3],1]-centri[a[2],1])**2
            ATA[a[3]][1,1] += (centri[a[2],1]-centri[a[3],1])**2
            
            b[a[2]][0,0] += (centri[a[3],0]-centri[a[2],0])*(phiIni[a[3]]-phiIni[a[2]])
            b[a[2]][1,0] += (centri[a[3],1]-centri[a[2],1])*(phiIni[a[3]]-phiIni[a[2]])
            
            b[a[3]][0,0] += (centri[a[2],0]-centri[a[3],0])*(phiIni[a[2]]-phiIni[a[3]])
            b[a[3]][1,0] += (centri[a[2],1]-centri[a[3],1])*(phiIni[a[2]]-phiIni[a[3]])
        
        if a[3]==-1 and bcdata[a[4]][0] == 'Dirichlet':
            ATA[a[2]][0,0] += (mAx[u]-centri[a[2],0])**2
            ATA[a[2]][0,1] += (mAx[u]-centri[a[2],0])*(mAy[u]-centri[a[2],1])
            ATA[a[2]][1,0] += (mAx[u]-centri[a[2],0])*(mAy[u]-centri[a[2],1])
            ATA[a[2]][1,1] += (mAy[u]-centri[a[2],1])**2
            
            b[a[2]][0,0] += (mAx[u]-centri[a[2],0])*(bcdata[a[4]][1](mAx[u],mAy[u])-phiIni[a[2]])
            b[a[2]][1,0] += (mAy[u]-centri[a[2],1])*(bcdata[a[4]][1](mAx[u],mAy[u])-phiIni[a[2]])
        
#        if a[3]==-1 and bcdata[a[4]][0] == 'Neumann':
            
            
        u+=1     
        
    gradPhi = np.linalg.solve(ATA,b)
    
    return gradPhi

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
    global PNKSI, PNN,A,ETA,KSI,deta,vel,X,Y,solU,solV,tim2,flag,solPhip,fSx,fSy,F,AA,aireTri,uF
    
    varInit(numelx,numely)
    normGen()
    KSI,DKSI = ksiGen()
    ETA,deta = etaGen()
    aireTri = airTri(tri)
    
    print('\nComputing for '+str(numelx)+'x'+str(numely)+' mesh')
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    
    x,y,uu,vv = sp.symbols('x y uu vv')
    uu = (2*x**2 - x**4 - 1)*(y-y**3)
    vv = -(2*y**2 - y**4 - 1)*(x-x**3)
    mu = 0.01
    rho = 1
    
    convx = sp.simplify(sp.diff(uu*uu,x)+sp.diff(vv*uu,y))
    diffx = sp.simplify(sp.diff(sp.diff(uu,x),x)+sp.diff(sp.diff(uu,y),y))
    Sourcex = convx - mu*diffx
    
    convy = sp.simplify(sp.diff(uu*vv,x)+sp.diff(vv*vv,y))
    diffy = sp.simplify(sp.diff(sp.diff(vv,x),x)+sp.diff(sp.diff(vv,y),y))
    Sourcey = convy - mu*diffy
    
    fSx = sp.lambdify([x,y],Sourcex,'numpy')
    fSy = sp.lambdify([x,y],Sourcey,'numpy')
    
    X = centri[:,0]
    Y = centri[:,1]
    
    Sx = fSx(X,Y)
    Sy = fSy(X,Y)

    qx = Sx
    qy = Sy
    
    P = np.sqrt(np.square(qx)+np.square(qy))
      
    if pltTriang:
        ax1.quiver(mAx,mAy,nx,ny,width=0.006)
        ax1.scatter(centri[:,0],centri[:,1],marker = '.',c='b')
        ax1.quiver(mAx,mAy,KSI[:,0],KSI[:,1],width=0.003,color='g')
        ax1.quiver(mAx,mAy,ETA[:,0],ETA[:,1],width=0.004,color='r')
        plt.show()
        plt.show()
    
    A = np.zeros((np.size(tri,0),np.size(tri,0)))
    
    PNKSI = np.zeros((np.size(are,0)))
    PNN = np.zeros((np.size(are,0)))
    D = np.zeros((np.size(are,0)))
    
    F = np.zeros((np.size(are,0)))
    
    up = np.zeros((np.size(tri,0)))
    vp = np.zeros((np.size(tri,0)))
    
    upp= np.zeros((np.size(tri,0)))
    vpp = np.zeros((np.size(tri,0)))
    alpha = 0.6
    
    
    from tqdm import tqdm
    pbar = tqdm(total=100)
    
    tim2=time.time()
    
    itt = 0
    while itt < 1:
        
        pbar.set_description('Processing iteration '+ str(itt ))
         
        if sum(qx) == 0 and sum(qy)==0: 
            bx = np.zeros((np.size(tri,0)))
            by = np.zeros((np.size(tri,0)))
        else:
            bx = np.ones((np.size(tri,0)))*aireTri*qx
            by = np.ones((np.size(tri,0)))*aireTri*qy
        
        Fx = rho*up
        Fy = rho*vp
        
        A = np.zeros((np.size(tri,0),np.size(tri,0)))
        u = 0;
        for i in are:
            ndot = np.dot(N[u,:],N[u,:])
            PNN[u] = ndot
            PNKSI[u] = np.dot(N[u,:],KSI[u,:])
            lmbda = ((mu*deta[u])/(DKSI[u]))*(ndot/PNKSI[u])
            D[u] = lmbda

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
                 
        u = 0;  
        for i in are:
            if (i[3] == -1 and bcdata_u[i[4]][0]=='Dirichlet'):
                bx[i[2]] += D[u]*bcdata_u[i[4]][1](mAx[u],mAy[u]) + max(0,-F[u])*bcdata_u[i[4]][1](mAx[u],mAy[u])
                
            if (i[3] == -1 and bcdata_v[i[4]][0]=='Dirichlet'):
                 by[i[2]] += D[u]*bcdata_v[i[4]][1](mAx[u],mAy[u]) + max(0,-F[u])*bcdata_v[i[4]][1](mAx[u],mAy[u])    
            u += 1
        
        solNodex = np.zeros((np.size(xx)))
        solNodey = np.zeros((np.size(yy)))
    
        flag = True
        it = 0
        solPhipx = np.zeros((np.size(tri,0)))
        solPhipy = np.zeros((np.size(tri,0)))
        solU = np.zeros((np.size(tri,0)))
        solV = np.zeros((np.size(tri,0)))
        
        while flag and it < 150:
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
            
            A_sp = sprs.csr_matrix(A)
            from scipy.sparse.linalg import spsolve
            
#            solU = np.linalg.solve(A,B1)
            solU = spsolve(A_sp,B1)
#            solV = np.linalg.solve(A,B2)
            solV = spsolve(A_sp,B2)
        
            solNodex = nodal_interpolation(solU,'x')
            solNodey = nodal_interpolation(solV,'y')
            
            if max(np.abs(solU - solPhipx)) < 1e-8 and max(np.abs(solV - solPhipy)) < 1e-8:
                flag = False
                
            it+= 1
            
        upp = np.copy(up)
        vpp = np.copy(vp)
        
        up = np.copy(solU)
        vp = np.copy(solV)
        
        if itt > 1:
            up = alpha*up + (1-alpha)*upp
            vp = alpha*vp + (1-alpha)*vpp
        
        pbar.update(1)
        itt += 1
        
    pbar.close()
    tim2 = time.time()-tim2
        
    
    UU = np.zeros((np.size(tri,0),2))
    UU[:,0] = solU
    UU[:,1] = solV
    
    uF = np.zeros((np.size(are,0)))
    
    gradPhi = gradLS(P)
    
    u=0
    for a in are:
        if a[3]!=-1:
            dVP = aireTri[a[2]]*1
            dVA = aireTri[a[3]]*1
            aP = A[a[2],a[2]]
            aA = A[a[3],a[3]]
            
            diffP = (1/2)*(dVP/aP + dVA/aA)*(P[a[2]]-P[a[3]])*(1/DKSI[u])
            diffgradP = np.transpose((1/2)*((dVP/aP)*gradPhi[a[2]] + (dVA/aA)*gradPhi[a[3]]))
            
#            uF[u] = np.dot((1/2)*(nU[a[2]]+nU[a[3]]),N[u]) + diffP + np.dot(diffgradP,KSI[u,:])
            uF[u] = np.dot([(1/2)*(UU[a[2],:]+UU[a[3],:])],N[u]) + diffP + np.dot(diffgradP,KSI[u,:])
            
        if (a[3]==-1 and bcdata_u[a[4]][0]=='Dirichlet' and bcdata_v[a[4]][0] =='Dirichlet'):
            uF[u] = np.dot(np.array([bcdata_u[a[4]][1](mAx[u],mAy[u]),bcdata_v[a[4]][1](mAx[u],mAy[u])]),N[u,:])
        
        if (a[3]==-1 and bcdata_u[a[4]][0]=='Neumann' and bcdata_v[a[4]][0]=='Neumann'):
            uF[u] = np.dot(UU[a[2]],N[u]) + (dVP/aP)*(P[a[2]]-bcdata[a[4]][1](mAx[u],mAy[u]))/DKSI[u] + (dVP/aP)*np.dot(gradPhi[a[2]],KSI[u,:])
        u+=1

    print('=========================================================================')
    print(' Total running time: '+str(round(tim2,2)))
    print(' Total number of iterations: '+str(itt)+'  Tolerance: 1e-8')
    print('=========================================================================')
    
    
    return A,bx,by

#%%
    
pltTriang = not True
upwind = not True

msolv2D(2,2)

X = centri[:,0]
Y = centri[:,1]

P = fP(X,Y)

gradPhi = gradLS(P).transpose(2,0,1).reshape(-1,2)
#print(gradPhi)

gradPhi_mod = np.sqrt(np.square(gradPhi[:,0])+np.square(gradPhi[:,1]))

x,y = sp.symbols('x y')
Gradx = sp.simplify(sp.diff(funPress,x))
Grady = sp.simplify(sp.diff(funPress,y))

fgradx = np.vectorize(sp.lambdify([x,y],Gradx,'numpy'))
fgrady = np.vectorize(sp.lambdify([x,y],Grady,'numpy'))

gradx_ana = fgradx(X,Y)
grady_ana = fgradx(X,Y)

modGrad = np.sqrt(np.square(gradx_ana)+np.square(grady_ana))

Sx = fSx(X,Y)
Sy = fSy(X,Y)

U = fu(X,Y)
V = fv(X,Y)

VEL = np.sqrt(np.square(U)+np.square(V))
S = np.sqrt(np.square(Sx)+np.square(Sy))

solVelocity = np.sqrt(np.square(solU)+np.square(solV))
errTri = np.abs(solVelocity-VEL)
errGrad = np.abs(gradPhi_mod - modGrad)

fig1, ax1 = plt.subplots()
tcf = ax1.tripcolor(MTri1, facecolors = VEL,edgecolor = 'k',cmap = 'viridis')
fig1.colorbar(tcf)
ax1.quiver(centri[:,0],centri[:,1],U,V,width=0.003)
ax1.set_title('Champs de vitesse donné \n (Analytique)')
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y Axis')
ax1.axes.set_aspect('equal')
plt.show()

fig2, ax2 = plt.subplots()
tcf = ax2.tripcolor(MTri1, facecolors=solVelocity,edgecolor = 'k',cmap = 'viridis')
fig2.colorbar(tcf)
ax2.quiver(centri[:,0],centri[:,1],solU,solV,width=0.003)
ax2.set_title('Solution numérique \n sur triangulation')
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y Axis')
ax2.axes.set_aspect('equal')
plt.show()

fig4, ax4 = plt.subplots()
tcf = ax4.tripcolor(MTri1, facecolors=errTri,edgecolor = 'k',cmap = 'viridis')
fig4.colorbar(tcf)
ax4.set_title('Erreur aux triangles - Vitesses')
ax4.set_xlabel('X axis')
ax4.set_ylabel('Y Axis')
ax4.axes.set_aspect('equal')
plt.show()


fig5, ax5 = plt.subplots()
tcf = ax5.tripcolor(MTri1, facecolors=modGrad,edgecolor = 'k',cmap = 'viridis')
fig5.colorbar(tcf)
ax5.set_title('Module du gradient de pression - Analytique')
ax5.set_xlabel('X axis')
ax5.set_ylabel('Y Axis')
ax5.axes.set_aspect('equal')
plt.show()

fig6, ax6 = plt.subplots()
tcf = ax6.tripcolor(MTri1, facecolors=gradPhi_mod,edgecolor = 'k',cmap = 'viridis')
fig6.colorbar(tcf)
ax6.set_title('Module du gradient de pression - Numérique \n (Least squares method)')
ax6.set_xlabel('X axis')
ax6.set_ylabel('Y Axis')
ax6.axes.set_aspect('equal')
plt.show()

uFx = uF * N[:,0]
uFy = uF * N[:,1]

fig7, ax7 = plt.subplots()
ax7.quiver(mAx,mAy,uFx,uFy,width=0.006)
ax7.set_aspect('equal')
ax7.triplot(MTri1, 'k.-', lw=1)
ax7.set_title('Triangulation Delaunay du domaine')
ax7.set_adjustable("datalim")
ax7.set_xlim(xmin - 0.1*(xmax-xmin), xmax + 0.1*(xmax-xmin))

#%% Orde de convergence (Centré)
#
#nraff = np.array([2,4,8,16,32])
#NORME = np.zeros((np.size(nraff)))
#ordre = np.zeros((np.size(nraff))-1)
#
#for ii in range(0,np.size(nraff)):
#    
#    nelm = nraff[ii]
#    
#    msolv2D(nelm,nelm)
#    
#    SolAna = fVel(X,Y)
#    solVelocity = np.sqrt(np.square(solU)+np.square(solV))
#    
##    errTri = np.abs(solVelocity-SolAna)
#    errGrad = np.abs(gradPhi_mod - modGrad)
#    normL2 = np.sqrt((1/np.size(solVelocity))*np.sum(np.square(errTri)));
#    
#    NORME[ii]=normL2
#    
#
#for n in range(1,np.size(nraff)):
#    ordre[n-1] = np.log(NORME[n-1]/NORME[n])/np.log(2)
#    
#print('Norme L2: ',NORME)
#print('Ordre de convergence: ',ordre)
#
#L =xmax-xmin
#
#fig=plt.figure()
#plt.loglog(L/nraff,NORME,'-ko')
#plt.title('Ordre de convergence pour le gradient- LAPP5 \n Upwind:'+str(upwind)+' | Pente = ' + str(round(np.average(ordre),2)))
#plt.ylabel('$Erreur$')
#plt.xlabel('$\Delta h$')
#ax = fig.add_subplot(111)
#ax.grid(b=True, which='minor', color='grey', linestyle='--')
#ax.grid(b=True, which='major', color='k', linestyle='-')
#
#print('La vitesse aux arètes est donnée par le vecteur "uF" suivant:')
#print(uF)