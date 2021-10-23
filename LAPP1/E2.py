# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 09:35:01 2020

@author: stsve
"""

import numpy as np
import matplotlib.pyplot as plt

#--------------------------------------------------------#
#              Initiation des variables                  #
#--------------------------------------------------------#

#Nombre de noeuds de discretisation
n = 5;
L =0.02;    # Longeur en m
dx = L/n;

Ta = 100;
Tb = 200;
q = 1000*10**3;


A = np.zeros((n,n));
b = np.ones((n,1));

k = 0.5;        #coefficient de diffusion thermique
S = 1;          #section de la tige

lmbda = k*S/dx;

#--------------------------------------------------------#
#         Construction du système  (Ax = b)              #
#--------------------------------------------------------#

#Premiere ligne de A
A[0,0]=3*lmbda;
A[0,1]=-lmbda;

#Derniere ligne de A
A[n-1,n-1]=3*lmbda;
A[n-1,n-2]=-lmbda;

for i in range(1,n-1):
    A[i,i+1] = -lmbda;
    A[i,i] = 2*lmbda;
    A[i,i-1] = -lmbda;
    

b=b*(q*S*dx);    
b[0,0]=b[0,0]+2*lmbda*Ta
b[n-1,0]=b[n-1,0]+2*lmbda*Tb;

#--------------------------------------------------------#
#             Solution du système  (Ax = b)              #
#--------------------------------------------------------#

T = np.linalg.solve(A, b);

#--------------------------------------------------------#
#         Post Traitement pour affichage                 #
#--------------------------------------------------------#

x = np.zeros((1,n+2));
x2= np.linspace(dx/2,(L-dx/2),n)
x[0,1:n+1]=x2
x[0,n+1]=L;

Tsol = np.zeros((1,n+2));
Tsol[0,1:n+1]=T.reshape((1,-1));
Tsol[0,0]=Ta;
Tsol[0,n+1]=Tb;

plt.plot(x[0,0:n+2],Tsol[0,0:n+2],'sk',label="Solution numérique")

#--------------------------------------------------------#
#                Erreur & Convergence                    #
#--------------------------------------------------------#

x_ex = np.linspace(0,L,100);
T_ex = ( ((Tb-Ta)/L)+q/(2*k) * (L-x_ex) )*x_ex + Ta;
plt.plot(x_ex,T_ex,'-b',label="Solution analytique")
plt.legend()

T_a = ( ((Tb-Ta)/L)+q/(2*k) * (L-x2) )*x2 + Ta;

err = np.divide(np.abs(Tsol[0,1:n+1] - ((((Tb-Ta)/L)+q/(2*k) * (L-x2) )*x2 + Ta)),T_a);
norm_err = np.sqrt((1/np.size(Tsol[0,1:n+1]))*np.sum(np.square(err)));


#--------------------------------------------------------#
#                Ordre de convergence                    #
#--------------------------------------------------------#
    
# Boucle pour recueuil de l'erreur en foction du maillage

p=np.array([10,100,1000]);
pp = np.zeros((1,3));

ordre = np.zeros((1,2));

for u in range(3):
    n = p[u];
    
    dx = L/n;
    A = np.zeros((n,n));
    b = np.ones((n,1));
    lmbda = k*S/dx;
    A[0,0]=3*lmbda;
    A[0,1]=-lmbda;
    A[n-1,n-1]=3*lmbda;
    A[n-1,n-2]=-lmbda;

    for i in range(1,n-1):
        A[i,i+1] = -lmbda;
        A[i,i] = 2*lmbda;
        A[i,i-1] = -lmbda;
    

    b=b*(q*S*dx);    
    b[0,0]=b[0,0]+2*lmbda*Ta
    b[n-1,0]=b[n-1,0]+2*lmbda*Tb;
    T = np.linalg.solve(A, b);
    
    x = np.zeros((1,n+2));
    x2= np.linspace(dx/2,(L-dx/2),n)
    x[0,1:n+1]=x2
    x[0,n+1]=L;
    
    Tsol = np.zeros((1,n+2));
    Tsol[0,1:n+1]=T.reshape((1,-1));
    Tsol[0,0]=Ta;
    Tsol[0,n+1]=Tb;
    x_ex = np.linspace(0,L,100);
    T_ex = ( ((Tb-Ta)/L)+q/(2*k) * (L-x_ex) )*x_ex + Ta;
    T_a = ( ((Tb-Ta)/L)+q/(2*k) * (L-x2) )*x2 + Ta;

    err = np.divide(np.abs(Tsol[0,1:n+1] - ((((Tb-Ta)/L)+q/(2*k) * (L-x2) )*x2 + Ta)),T_a);
    norm_err = np.sqrt((1/np.size(Tsol[0,1:n+1]))*np.sum(np.square(err)));
    pp[0,u] = norm_err;
    

for i in range(1,3):
    ordre[0,i-1]=np.log(pp[0,i-1]/pp[0,i])/np.log(p[i]/p[i-1])

fig=plt.figure()
plt.loglog(L/p,pp[0,0:3])
plt.ylabel('$Erreur$')
plt.xlabel('$\Delta h$')
ax = fig.add_subplot(111)
ax.grid(b=True, which='minor', color='grey', linestyle='--')
ax.grid(b=True, which='major', color='k', linestyle='-')

print('L''ordre de convergence observé est:')
print(ordre[0,1])
    