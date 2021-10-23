import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import mec6616
import numpy as np
import time
import sympy as sp
import scipy.sparse as sprs
from scipy.sparse.linalg import spsolve

#%%

def pressionCorrection(uF,P,A):
    global bcdataP,DKSI,dA
    
    ntri = np.size(tri,0)
    MP = np.zeros([ntri,ntri])
    BP = np.zeros(ntri)
    rho = 1
    uFnew = np.zeros(np.size(are,0))
    uFnew += uF
    
    u=0
    for a in are:
        if a[3] != 1:
            
            dvP = aireTri[a[2]]
            dvA = aireTri[a[3]]
            aP = A[a[2],a[2]]
            aA = A[a[3],a[3]]
            
            dfi = (1/2)*(dvP/aP + dvA/aA)/(DKSI[u])
            
            MP[a[2],a[2]] += rho*dfi*dA[u]
            MP[a[3],a[3]] += rho*dfi*dA[u]
            MP[a[2],a[3]] -= rho*dfi*dA[u]
            MP[a[3],a[2]] -= rho*dfi*dA[u]
            BP[a[2]] -= rho*uF[u]*dA[u]
            BP[a[3]] += rho*uF[u]*dA[u]
            
        elif (a== -1 and bcdataP[a[4]][0] == 'Sortie'):
            
            dvP = aireTri[a[2]]
            aP = A[a[2],a[2]]
            dfi = dvP/aP*(1/DKSI[u])
            
            MP[a[2],a[2]] += rho*dfi*dA[u]
            BP[a[2]] -= rho*uF[u]*dA[u]
        u+=1
    
    PP = np.linalg.solve(MP,BP)
        
    u=0
    for a in are:
        if a[3]!=-1:
            
            dvP = aireTri[a[2]]
            dvA = aireTri[a[3]]
            aP = A[a[2],a[2]]
            aA = A[a[3],a[3]] 
            dfi = (1/2)*(dvP/aP + dvA/aA)/(DKSI[u])
            uFnew[u] += dfi*(PP[a[2]]-PP[a[3]])
            
        elif (a[3]== -1 and bcdataP[a[4]][0]=='Sortie'):
            
            dvP = aireTri[a[2]]
            aP = A[a[2],a[2]]
            dfi = dvP/aP*(1/DKSI[u])
            
            uFnew[u] += dfi*PP[a[2]]
        u+=1
        
    return uFnew,PP