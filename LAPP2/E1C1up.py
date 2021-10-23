#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 16:32:03 2020

*** Approximation Upwind ***

@author: stefane sved (1569161)
"""

import numpy as np
import matplotlib.pyplot as plt

#--------------------------------------------------------#
#              Initiation des variables                  #
#--------------------------------------------------------#

n = 5;          # Nombre de noeuds de discretisation
L =1;           # Longeur en m
rho = 1;
u = 0.1;        # vitesse (0.1 | 2.5) [m/s]
Gamma=0.1;


dx = L/n;

Ga = 1;
Gb = 0;


A = np.zeros((n,n));
b = np.zeros((n,1));


lmbda = Gamma/dx;

#--------------------------------------------------------#
#         Construction du système  (Ax = b)              #
#--------------------------------------------------------#

#Premiere ligne de A
A[0,0]=(3*lmbda + rho*u);
A[0,1]=-lmbda;

#Derniere ligne de A
A[n-1,n-1]=(3*lmbda + rho*u);
A[n-1,n-2]=-(lmbda+rho*u);

for i in range(1,n-1):
    A[i,i+1] = -lmbda;
    A[i,i] = (2*lmbda + rho*u);
    A[i,i-1] = -(lmbda+rho*u);
    
   
b[0,0]=(2*lmbda+rho*u)*Ga;
b[n-1,0]=(2*lmbda)*Gb;
#--------------------------------------------------------#
#             Solution du système  (Ax = b)              #
#--------------------------------------------------------#

G = np.linalg.solve(A, b);

#--------------------------------------------------------#
#         Post Traitement pour affichage                 #
#--------------------------------------------------------#



x = np.zeros((1,n+2));
x2= np.linspace(dx/2,(L-dx/2),n)
x[0,1:n+1]=x2
x[0,n+1]=L;

Gsol = np.zeros((1,n+2));
Gsol[0,1:n+1]=G.reshape((1,-1));
Gsol[0,0]=Ga;
Gsol[0,n+1]=Gb;

plt.plot(x[0,0:n+2],Gsol[0,0:n+2],'--sk',label="Solution numérique")

#--------------------------------------------------------#
#                Erreur & Convergence                    #
#--------------------------------------------------------#

x_ex = np.linspace(0,L,100);
G_ex = (Gb-Ga) * (np.exp((rho*u*x_ex)/Gamma)-1)/( np.exp(rho*u*L/Gamma)-1) + Ga;


plt.plot(x_ex,G_ex,'-b',label="Solution analytique")
plt.legend()

G_analitic = (Gb-Ga) * (np.exp((rho*u*x2)/Gamma)-1)/( np.exp(rho*u*L/Gamma)-1) + Ga;

err = np.divide(np.abs(Gsol[0,1:n+1] - ((Gb-Ga) * (np.exp((rho*u*x2)/Gamma)-1)/( np.exp(rho*u*L/Gamma)-1) + Ga)),G_analitic);
norm_err = np.sqrt((1/np.size(Gsol[0,1:n+1]))*np.sum(np.square(err)));

   
    #--------------------------------------------------------#
    #                Ordre de convergence                    #
    #--------------------------------------------------------#
        
# Boucle pour recueuil de l'erreur en foction du maillage

p=np.array([10,100,1000]);
pp = np.zeros((1,3));

ordre = np.zeros((1,2));

for uu in range(3):
    n = p[uu];
    
    dx = L/n;
    A = np.zeros((n,n));
    b = np.zeros((n,1));
    lmbda = Gamma/dx;

    A[0,0]=3*lmbda + rho*u/2;
    A[0,1]=-(lmbda-rho*u/2);
    A[n-1,n-1]=3*lmbda - rho*u/2;
    A[n-1,n-2]=-(lmbda+rho*u/2);

    for i in range(1,n-1):
        A[i,i+1] = -(lmbda-rho*u/2);
        A[i,i] = 2*lmbda;
        A[i,i-1] = -(lmbda+rho*u/2);
    

    b[0,0]=(2*lmbda+rho*u)*Ga;
    b[n-1,0]=(2*lmbda-rho*u)*Gb;
    
    G = np.linalg.solve(A, b);
    
    
    x = np.zeros((1,n+2));
    x2= np.linspace(dx/2,(L-dx/2),n)
    x[0,1:n+1]=x2
    x[0,n+1]=L;
    
    Gsol = np.zeros((1,n+2));
    Gsol[0,1:n+1]=G.reshape((1,-1));
    Gsol[0,0]=Ga;
    Gsol[0,n+1]=Gb;
    x_ex = np.linspace(0,L,100);
    
    G_ex = (Gb-Ga) * (np.exp((rho*u*x_ex)/Gamma)-1)/( np.exp(rho*u*L/Gamma)-1) + Ga;
    G_analitic = (Gb-Ga) * (np.exp((rho*u*x2)/Gamma)-1)/( np.exp(rho*u*L/Gamma)-1) + Ga;

    err = np.divide(np.abs(Gsol[0,1:n+1] - ((Gb-Ga) * (np.exp((rho*u*x2)/Gamma)-1)/( np.exp(rho*u*L/Gamma)-1) + Ga)),G_analitic);
    norm_err = np.sqrt((1/np.size(Gsol[0,1:n+1]))*np.sum(np.square(err)));
    

    pp[0,uu] = norm_err;
        
    # Ordre de convergence
    
    for i in range(1,3):
        ordre[0,i-1]=np.log(pp[0,i-1]/pp[0,i])/np.log(p[i]/p[i-1])
    
fig=plt.figure()
plt.loglog(L/p,pp[0,0:3],'-o')
plt.ylabel('$Erreur$')
plt.xlabel('$\Delta h$')
ax = fig.add_subplot(111)
ax.grid(b=True, which='minor', color='grey', linestyle='--')
ax.grid(b=True, which='major', color='k', linestyle='-')

print('L''ordre de convergence observé est:')
print(np.average(ordre))
    