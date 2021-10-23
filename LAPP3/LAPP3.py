#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:11:36 2020

@author: stefanesved
"""

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import mec6616
import numpy as np

nx = 3;
ny = 3;
n_noeud = (nx+1)*(ny+1);

xmin = 0;
xmax = 2;
ymin = 0;
ymax = 1;

xx,yy,tri,are = mec6616.RectMesh(xmin,xmax,ymin,ymax,nx,ny)

#triangulation matplotlib.tri pour graphes
MTri1 = mtri.Triangulation(xx, yy,tri)   #objet Triangulation pour figures



fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
ax1.triplot(MTri1, 'ko-', lw=1)
ax1.set_title('triplot of Delaunay triangulation')

#trace la triangulation
fi2, ax2 = plt.subplots()
ax2.set_aspect('equal')
ax2.triplot(MTri1)

#annoter les noeuds
for i in range (0,xx.size):
    plt.annotate(str(i),(xx[i],yy[i]))
#annoter les triangles
for i in range (0,tri.shape[0]):
    xc = (xx[tri[i,0]]+xx[tri[i,1]]+xx[tri[i,2]])/3.0
    yc = (yy[tri[i,0]]+yy[tri[i,1]]+yy[tri[i,2]])/3.0    
    plt.annotate(str(i),(xc,yc))    
plt.show()
    
bcdata = (['Dirichlet',1],['Neumann',0],['Dirichlet',5],['Neumann',0])

Solx1 = mec6616.TriMoyenne(xx,yy,tri,are,bcdata)


#--------------------------------------------------------#
#               Matrice d'interpolation                  #
#--------------------------------------------------------#

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

solTri = Solx1.reshape((-1,1))
solTri = np.vstack((solTri,np.zeros((1,1))))

solNode = mat.dot(solTri)[0:n_noeud];      #Solution aux noeuds

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

print('La solution aux noeuds est:')
print(solNode.reshape((1,-1)))

xt = xT.reshape((1,-1))
yt = yT.reshape((1,-1))

solNode_flat = solNode.ravel()

fig3, ax3 = plt.subplots()
ax3.set_aspect('equal')
tcf = ax3.tricontourf(MTri1, solNode_flat)
fig3.colorbar(tcf)
ax3.tricontour(MTri1, solNode_flat, colors='k')
ax3.set_title('Contour plot of Delaunay triangulation')
plt.show()



#--------------------------------------------------------#
#               Calcul de la divergence                  #
#--------------------------------------------------------#



#--------------------------------------------------------#
#            Fonction calcul divergence                  #
#--------------------------------------------------------#
def divTri(Fx,Fy,are,tri,xx,yy):
    
    divT = np.zeros((np.size(tri,0),1))
    
    dxa = xx[are[:,1]]-xx[are[:,0]]
    dya = yy[are[:,1]]-yy[are[:,0]]
    
    nx=dya;
    ny=-dxa;
    
    u=0;
    for i in are:
        if i[3] == -1:
            divx = nx[u]*Fx;
            divy = ny[u]*Fy;
            divT[i[2]] = divT[i[2]] + (divx+divy)
        else:
            divx = nx[u]*Fx;
            divy = ny[u]*Fy;
            divT[i[2]] = divT[i[2]] + (divx + divy)
            divT[i[3]] = divT[i[3]] - (divx + divy)
        u=u+1;
    
    return divT
#--------------------------------------------------------#    
    
    

# Composantes du champs vectoriel constant
Fx=5;
Fy=2;

divT = np.zeros((np.size(tri,0),1))

# Calcul des longeurs des aretes
dxa = xx[are[:,1]]-xx[are[:,0]]
dya = yy[are[:,1]]-yy[are[:,0]]
xca = 0.5*(xx[are[:,0]]+xx[are[:,1]])
yca = 0.5*(yy[are[:,0]]+yy[are[:,1]])


# Vecteurs des normales pour chaque aretes 
nx=dya;
ny=-dxa;

u=0;
for i in are:
    if i[3] == -1:
        divx = nx[u]*Fx;
        divy = ny[u]*Fy;
        divT[i[2]] = divT[i[2]] + (divx+divy)
    else:
        divx = nx[u]*Fx;
        divy = ny[u]*Fy;
        divT[i[2]] = divT[i[2]] + (divx + divy)
        divT[i[3]] = divT[i[3]] - (divx + divy)
    u=u+1;


print('La divergence à chaque triangle est:')
print(divT)



    
    
    
