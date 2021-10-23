#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:54:28 2020

@author: stefanesved
"""

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import mec6616
import numpy as np

nx = 2;
ny = 2;
n_noeud = (nx+1)*(ny+1);

xmin = 0;
xmax = 2;
ymin = 0;
ymax = 1;

xx,yy,tri,are = mec6616.RectMesh(xmin,xmax,ymin,ymax,nx,ny)

#triangulation matplotlib.tri pour graphes
MTri1 = mtri.Triangulation(xx, yy,tri)   #objet Triangulation pour figures

#trace la triangulation
plt.triplot(MTri1)
plt.axes().set_aspect('equal')
#annoter les noeuds
for i in range (0,xx.size):
    plt.annotate(str(i),(xx[i],yy[i]))
#annoter les triangles
for i in range (0,tri.shape[0]):
    xc = (xx[tri[i,0]]+xx[tri[i,1]]+xx[tri[i,2]])/3.0
    yc = (yy[tri[i,0]]+yy[tri[i,1]]+yy[tri[i,2]])/3.0    
    plt.annotate(str(i),(xc,yc))   
#annoter les aretes - #numero d'arete-cf
for i in range (0,are.shape[0]):
    xc = (xx[are[i,0]]+xx[are[i,1]])/2.0
    yc = (yy[are[i,0]]+yy[are[i,1]])/2.0    
    plt.annotate(str(i)+'-'+str(are[i,4]),(xc,yc)) 
    
#Test de TriMoyenne sur maillage de RectMesh   
#codage des conditions limites - tuple de liste
bcdata = (['Dirichlet',1],['Neumann',0],['Dirichlet',5],['Neumann',0])

Solx1 = mec6616.TriMoyenne(xx,yy,tri,are,bcdata) 



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

solNoeud = mat.dot(solTri)[0:n_noeud];


xT = xx.reshape((-1,1))
yT = yy.reshape((-1,1))

boundx = np.array([xmin,xmax])
boundy = np.array([ymin,ymax])

for i in range(2):
    locx,_ = np.where(xT == boundx[i])
    for u in locx:
        if bcdata[i][0] == 'Dirichlet':
            solNoeud[u] = bcdata[i][1]
            
locx_min,_ = np.where(xT == boundx[0]);
locx_max,_ = np.where(xT == boundx[1]);
locy_min,_ = np.where(yT == boundy[0]);
locy_max,_ = np.where(yT == boundy[1]);

for i in locx_min:
    if bcdata[0][0] == 'Dirichlet':
        solNoeud[i]=bcdata[0][1]
        
for i in locx_max:
    if bcdata[2][0] == 'Dirichlet':
        solNoeud[i]=bcdata[2][1]
        
for i in locy_min:
    if bcdata[1][0] == 'Dirichlet':
        solNoeud[i]=bcdata[1][1]
        
for i in locy_max:
    if bcdata[3][0] == 'Dirichlet':
        solNoeud[i]=bcdata[3][1]


#knot_arr = np.vstack((locx_min,locx_max,locy_min,locy_max))
#
#
#for i in knot_arr:
#    u=0;
#    for ii in range(np.size(i)):
#        if bcdata[u][0] == 'Dirichlet':
#            solNoeud[i[ii]] = bcdata[u][1]
#        u=u+1;

#longueur des aretes
dxa = xx[are[:,1]]-xx[are[:,0]]
dya = yy[are[:,1]]-yy[are[:,0]]
xca = 0.5*(xx[are[:,0]]+xx[are[:,1]])
yca = 0.5*(yy[are[:,0]]+yy[are[:,1]])

#normales aux aretes; pointe vers l'élément à droite (Externe)
nx=dya;
ny=-dxa;
    
#tracage d'un champ de vecteur au milieu des sides, normale à l'arete

    
   #plot triangles with one color per triangle for solution Solx
fig1, ax1 = plt.subplots()
tcf = ax1.tripcolor(MTri1, facecolors=Solx1, edgecolors='k')
#tricontour
plt.quiver(xca,yca,nx,ny)
fig1.colorbar(tcf)
ax1.set_title('Contour plot on triangulation')
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y Axis')
ax1.axes.set_aspect('equal')
plt.show() 


print('Les valeurs aux noeuds sont:')
print(solNoeud)

#trace la triangulation
plt.triplot(MTri1)
plt.axes().set_aspect('equal')
#annoter les noeuds
for i in solNoeud.astype(int):
    plt.annotate(str(i),(xx[i],yy[i]))
    
    
#Composantes du champs vectoriel constant
Fx = 5;
Fy = 2;

dx=0;
dy=0;


u=0;
for i in are:
    if i[3] != -1:
        dx = dx + nx[u]*Fx;
        dx = dx-nx[u]*Fx;
        dy = dy + ny[u]*Fy;
        dy = dy-ny[u]*Fy
    
    dx = dx + nx[u]*Fx;
    dy = dy + ny[u]*Fy;
    u=u+1;
    





