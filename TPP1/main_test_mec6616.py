# -*- coding: utf-8 -*-
"""
@authors: Jean-Yves Trépanier et Eddy Petro

Programme principal pour tester les fonctions du module mec6616.py
Janvier 2020
5
"""
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import mec6616

#Test de RectMesh
xx,yy,tri,are = mec6616.RectMesh(0,2,0,1,2,2)

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

#plot triangles with one color per triangle for solution Solx
fig1, ax1 = plt.subplots()
tcf = ax1.tripcolor(MTri1, facecolors=Solx1, edgecolors='k')
fig1.colorbar(tcf)
ax1.set_title('Contour plot on triangulation')
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y Axis')
ax1.axes.set_aspect('equal')
plt.show()  





#
##Test de RectGmsh
#xx,yy,tri,are = mec6616.RectGmsh(0,2,0,1,0.5)
#
##triangulation matplotlib.tri pour graphes
#MTri2 = mtri.Triangulation(xx, yy,tri)   #objet Triangulation pour figures
#
##trace la triangulation
#plt.triplot(MTri2)
#plt.axes().set_aspect('equal')
##annoter les noeuds
#for i in range (0,xx.size):
#    plt.annotate(str(i),(xx[i],yy[i]))
##annoter les triangles
#for i in range (0,tri.shape[0]):
#    xc = (xx[tri[i,0]]+xx[tri[i,1]]+xx[tri[i,2]])/3.0
#    yc = (yy[tri[i,0]]+yy[tri[i,1]]+yy[tri[i,2]])/3.0    
#    plt.annotate(str(i),(xc,yc))   
##annoter les aretes - #numero d'arete-cf
#for i in range (0,are.shape[0]):
#    xc = (xx[are[i,0]]+xx[are[i,1]])/2.0
#    yc = (yy[are[i,0]]+yy[are[i,1]])/2.0    
#    plt.annotate(str(i)+'-'+str(are[i,4]),(xc,yc))
#    
##longueur des aretes
#dxa = xx[are[:,1]]-xx[are[:,0]]
#dya = yy[are[:,1]]-yy[are[:,0]]
#xca = 0.5*(xx[are[:,0]]+xx[are[:,1]])
#yca = 0.5*(yy[are[:,0]]+yy[are[:,1]])
#
##normales aux aretes; pointe vers l'élément à droite (Externe)
#nx=dya;
#ny=-dxa;
#
##tracage d'un champ de vecteur au milieu des sides, normale à l'arete
#plt.quiver(xca,yca,nx,ny)
#plt.show
#
##Test de AirfoilGmsh
#xx,yy,tri,are = mec6616.AirfoilGmsh('naca0012.txt',4,0.1,3)
#
##triangulation matplotlib.tri pour graphes
#MTri3 = mtri.Triangulation(xx, yy,tri)   #objet Triangulation pour figures
#
##trace la triangulation
#plt.figure()
#plt.triplot(MTri3)
#plt.axes().set_aspect('equal')
#plt.show
#
##Test de TriMoyenne sur maillage de AirfoilGmsh   
##codage des conditions limites - tuple de liste
#bcdata = (['Dirichlet',0],['Dirichlet',1],['Dirichlet',2],\
#          ['Dirichlet',3],['Dirichlet',4] )
#
#Solx3 = mec6616.TriMoyenne(xx,yy,tri,are,bcdata) 
#
##plot triangles with one color per triangle for solution Solx
#fig4, ax4 = plt.subplots()
#tcf = ax4.tripcolor(MTri3, facecolors=Solx3, edgecolors='k')
#fig4.colorbar(tcf)
#ax4.set_title('Contour plot on triangulation')
#ax4.set_xlabel('X axis')
#ax4.set_ylabel('Y Axis')
#ax4.axes.set_aspect('equal')
#plt.show()
#
#
##Test de BackstepGmsh
#xx,yy,tri,are = mec6616.BackstepGmsh(1,2,5,2,0.5)
#
##triangulation matplotlib.tri pour graphes
#MTri3 = mtri.Triangulation(xx, yy,tri)   #objet Triangulation pour figures
#
##trace la triangulation
#plt.triplot(MTri3)
#plt.axes().set_aspect('equal')
#plt.show()
#
##Test de CircleGmsh
#xx,yy,tri,are = mec6616.CircleGmsh(0,4,0,1,0.2,0.25,0.125)
#
##triangulation matplotlib.tri pour graphes
#MTri3 = mtri.Triangulation(xx, yy,tri)   #objet Triangulation pour figures
#
##trace la triangulation
#plt.triplot(MTri3)
#plt.axes().set_aspect('equal')
#plt.show()
#
##Test de TriMoyenne sur maillage de CircleGmsh   
##codage des conditions limites - tuple de liste
#bcdata = (['Dirichlet',0],['Dirichlet',1],['Dirichlet',2],\
#          ['Dirichlet',3],['Dirichlet',4] )
#
#Solx3 = mec6616.TriMoyenne(xx,yy,tri,are,bcdata) 
#
##plot triangles with one color per triangle for solution Solx
#fig6, ax6 = plt.subplots()
#tcf = ax6.tripcolor(MTri3, facecolors=Solx3, edgecolors='k')
#fig6.colorbar(tcf)
#ax6.set_title('Contour plot on triangulation')
#ax6.set_xlabel('X axis')
#ax6.set_ylabel('Y Axis')
#ax6.axes.set_aspect('equal')
#plt.show()
#
## Ecriture avec meshio d'un format vtk pour Paraview
#
#filename='CircleTest'
#NodeDataDict={'abcisse':xx,'ordonnee':yy}
#TriDataDict={'Solx3':Solx3}
#mec6616.meshiowritevtk(xx,yy,tri,NodeDataDict,TriDataDict,filename)
#



