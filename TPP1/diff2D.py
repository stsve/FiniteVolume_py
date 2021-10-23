import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import mec6616
import numpy as np

def intNode(are,tri,xx,yy,solPhi,bcdata,n_node,xmin,xmax,ymin,ymax):
    
    coord = [];

    for i in range (np.size(tri,0)+1):
        x,_ = np.where(tri == i)
        x = x.reshape((-1,1))
        coord.append(x)
        
    mat = np.zeros((np.size(tri,0)+1,np.size(tri,0)+1));
    
    col = 0;
    
    for c in coord:
        for l in c:
            mat[col,l]=1/np.size(c)   
        col=col+1;
    
    solTri = solPhi.reshape((-1,1))
    solTri = np.vstack((solTri,np.zeros((1,1))))
    
    solNode = mat.dot(solTri)[0:n_node];       #Solution aux noeuds
    
    
    xT = xx.reshape((-1,1))
    yT = yy.reshape((-1,1))
    
    boundx = np.array([xmin,xmax])
    boundy = np.array([ymin,ymax])
    
    nodex_min,_ = np.where(xT == boundx[0]);
    nodex_max,_ = np.where(xT == boundx[1]);
    nodey_min,_ = np.where(yT == boundy[0]);
    nodey_max,_ = np.where(yT == boundy[1]);
    
    
    for i in nodex_min:
        if bcdata[0][0] == 'Dirichlet':
            solNode[i]=bcdata[0][1]
            
    for i in nodex_max:
        if bcdata[2][0] == 'Dirichlet':
            solNode[i]=bcdata[2][1]
            
    for i in nodey_min:
        if bcdata[1][0] == 'Dirichlet':
            solNode[i]=bcdata[1][1]
            
    for i in nodey_max:
        if bcdata[3][0] == 'Dirichlet':
            solNode[i]=bcdata[3][1]
    
    solNode_flat = solNode.ravel()
    
    return solNode_flat

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

# ---------------------------   *****************  ----------------------------

# %%
#--------------------------------------------------------#
#     Dimmenssion du domaine  &   Discrétisation         #
#--------------------------------------------------------#

divx = 20                      # Nombre de divisions en x
divy = 20                    # Nombre de divisions en y
n_node = (divx+1)*(divy+1);     # Nombre total de noeuds

xmin = 0;                       # Limite inférieure du domaine en x
xmax = 0.02;                       # Limite supérieure du domaine en x
ymin = 0;                       # Limite inférieure du domaine en y
ymax = 0.01;                       # Limite supérieure du domaine en y

k = 0.5
q = 1000000;

#--------------------------------------------------------#
#              Triangulation (Delaunay)                  #
#--------------------------------------------------------#

xx,yy,tri,are = mec6616.RectMesh(xmin,xmax,ymin,ymax,divx,divy)
MTri1 = mtri.Triangulation(xx, yy,tri)

fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
ax1.triplot(MTri1, 'k.-', lw=1)
ax1.set_title('Triangulation Delaunay du domaine')
ax1.set_adjustable("datalim")
ax1.set_xlim(xmin - 0.1*(xmax-xmin), xmax + 0.1*(xmax-xmin))
#ax2.set_ylim(1e-1, 1e3)

#   annoter les noeuds
for i in range (0,xx.size):
    plt.annotate(str(i),(xx[i],yy[i]))
#   annoter les triangles
for i in range (0,tri.shape[0]):
    xc = (xx[tri[i,0]]+xx[tri[i,1]]+xx[tri[i,2]])/3.0
    yc = (yy[tri[i,0]]+yy[tri[i,1]]+yy[tri[i,2]])/3.0    
    plt.annotate(str(i),(xc,yc))    
plt.show()


#--------------------------------------------------------#
#                   Conditions limites                   #
#--------------------------------------------------------#

bcdata = (['Neumann',0],['Neumann',0],['Dirichlet',200],['Neumann',0])

#--------------------------------------------------------#
#        Generation des composantes n, ksi, eta          #
#--------------------------------------------------------#

dxA = xx[are[:,1]]-xx[are[:,0]]
dyA = yy[are[:,1]]-yy[are[:,0]]
mAx = 0.5*(xx[are[:,0]]+xx[are[:,1]]) # Coordonnées x du milieu sur chaque arete
mAy = 0.5*(yy[are[:,0]]+yy[are[:,1]]) # Coordonnées y du milieu sur chaque arete

dA = np.sqrt(np.square(dxA)+np.square(dyA))


# Calcul de l'aire d'un triangle
aireTri = airTri(tri)

# ------------

nx = dyA/dA;
ny = -dxA/dA;

plt.quiver(mAx,mAy,nx,ny,width=0.006)
plt.show()

deta = dA

centri = np.zeros((np.size(tri,0),2))

u = 0;
for t in tri:
    centri[u,0] = np.sum(xx[t])/3
    centri[u,1] = np.sum(yy[t])/3
    u += 1
    
plt.scatter(centri[:,0],centri[:,1],marker = '.',c='b')

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

# trace les composantes ksi pour les aretes internes seulement
#plt.quiver(mAx[ti[0][0]:ti[0][-1]+1],mAy[ti[0][0]:ti[0][-1]+1],xksi,yksi,width=0.003,color='g')
#plt.show()

plt.quiver(mAx,mAy,Xksi,Yksi,width=0.003,color='g')
plt.show()

## eta des aretes internes
#etax = dxA[ti[0][0]:ti[0][-1]]/dA[ti[0][0]:ti[0][-1]]
#etay = dyA[ti[0][0]:ti[0][-1]]/dA[ti[0][0]:ti[0][-1]]

delta_ksi = np.sqrt(np.square(Xksi)+np.square(Yksi))

ETA = np.zeros((np.size(dxA,0),2))
etax = dxA/dA
etay = dyA/dA

ETA[:,0] = etax
ETA[:,1] = etay


## trace les composantes eta internes 
#plt.quiver(mAx[ti[0][0]:ti[0][-1]+1],mAy[ti[0][0]:ti[0][-1]+1],etax,etay,width=0.004,color='r')

plt.quiver(mAx,mAy,etax,etay,width=0.004,color='r')
plt.show()

N = np.zeros((np.size(nx),2))
N[:,0] = nx
N[:,1] = ny


#--------------------------------------------------------#
#         Construction du système matriciel              #
#--------------------------------------------------------#

A = np.zeros((np.size(tri,0),np.size(tri,0)))

if q != 0:   
    b = np.ones((np.size(tri,0)))*q*aireTri
else:
    b = np.zeros((np.size(tri,0)))

u = 0;
for i in are:
    ndot = np.dot(N[u,:],N[u,:])
    pnksi = np.dot(N[u,:],KSI[u,:])
    lmbda = ((k*deta[u])/(DKSI[u]))*(ndot/pnksi)
      
#    if bcdata[i[4]][0] != 'Neumann':
    A[are[u,2],are[u,2]] += lmbda
    
    if are[u,3] != -1:
        A[are[u,2],are[u,3]] += -lmbda
        A[are[u,3],are[u,2]] += -lmbda
        A[are[u,3],are[u,3]] += lmbda
    
    u +=1
    
#A *= -1
    
u = 0;  
for i in are:
    ndot = np.dot(N[u,:],N[u,:])
    pnksi = np.dot(N[u,:],KSI[u,:])
    lmbda = ((k*deta[u])/(DKSI[u]))*(ndot/pnksi)
    
    if i[3] ==-1 and bcdata[i[4]][0] == 'Dirichlet':
        b[i[2]] += lmbda * bcdata[i[4]][1]
        
    u += 1
    

solNode = np.zeros((n_node))

bt = b

it = 1
while it < 100:
    
    S = np.zeros((np.size(b)))
    
    u = 0;
    for i in are:
        
        pksieta = np.dot(KSI[u,:],ETA[u,:])
        pnksi = np.dot(N[u,:],KSI[u,:])
        Scd = -(((k*dA[u])/deta[u]) * (pksieta/pnksi))*(solNode[i[1]]-solNode[i[0]])
          
        if i[3] == -1:
            S[i[2]] += Scd
        else:
            S[i[2]] += Scd
            S[i[3]] += -Scd
        
        u += 1
#        print(Scd)
    
    B = b + S;     
    print('B avec Sd:',B)
    
    solPhi = np.linalg.solve(A,B)
    solNode = intNode(are,tri,xx,yy,solPhi,bcdata,n_node,xmin,xmax,ymin,ymax)
    
    print('iter:',it)
    print('b:',b)
    print('-----------')
    print('Solphi',solPhi)

    it+= 1
   
#    
#solPhi = np.linalg.solve(A,b)

fig2, ax2 = plt.subplots()
tcf = ax2.tripcolor(MTri1, facecolors=solPhi, edgecolors='k')
fig2.colorbar(tcf)
ax2.set_title('Contour plot on triangulation')
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y Axis')
ax2.axes.set_aspect('equal')
plt.show()  
    
    



