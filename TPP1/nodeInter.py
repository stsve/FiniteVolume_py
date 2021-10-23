#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:53:24 2020

@author: stefanesved
"""

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
    
    solNode = mat.dot(solTri)[0:n_node];      #Solution aux noeuds
    
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
    
    return solNode