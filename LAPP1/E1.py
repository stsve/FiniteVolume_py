# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:26:12 2020

@author: stsve
"""

import numpy as np
import matplotlib.pyplot as plt

#--------------------------------------------------------#
#              Initiation des variables                  #
#--------------------------------------------------------#


n = 5;                  #Nombre de noeuds de discrétisation


L =0.5;
dx = L/n;
Ta = 100;
Tb = 500;


A = np.zeros((n,n));
b = np.zeros((n,1));

k = 1000;               #coefficient de diffusion thermique
S = 10*10**(-3);        #section de la tige

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
    
print('La matrice du système est la suivante:')
print(A)
    
b[0,0]=2*lmbda*Ta;
b[n-1,0]=2*lmbda*Tb;

print('Le vecteur b du système Ax=b est le suivant:')
print(b)
#--------------------------------------------------------#
#             Resolution du système  (Ax = b)            #
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

fig = plt.figure()
plt.plot(x[0,0:n+2],Tsol[0,0:n+2],'sk',label="Solution numérique")

#--------------------------------------------------------#
#                Erreur & Convergence                    #
#--------------------------------------------------------#

x_ex = np.linspace(0,L,100);
T_ex = 800*x_ex + 100;
plt.plot(x_ex,T_ex,'-b',label="Solution analytique")
plt.legend()

T_a = 800*x2 + 100;

err = np.divide(np.abs(Tsol[0,1:n+1]-(800*x2 + 100)),T_a);
norm_err = np.sqrt(np.sum(np.square(err)));