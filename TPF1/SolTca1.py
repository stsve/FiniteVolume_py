# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:00:03 2020

@author: stsve
"""

import numpy as np
import matplotlib.pyplot as plt
#import os


H=0.05
L = 0.1
k = 0.4
Tinf = 25
q = 135300
h = 60

N = np.array([1,2,3,4,5,6,7,8,9,10])
eigV = N*(np.pi) - (np.pi/2) 
#eigV =(2*N -1)*(np.pi/2)

x = np.linspace(0,0.1,201);
y = np.linspace(0,0.05,201);


Bi = (h*H)/k;
a = L/H;

grid = np.meshgrid(x,y)

series3 = [[],[],[],[],[],[],[],[],[],[]]

for n in range(1,np.size(N,0)+1):
    series3[n-1] = ((-1)**n/eigV[n-1]**3) * ((np.cosh((eigV[n-1]*grid[1])/(a*H))*np.cos(eigV[n-1]*(grid[0]/L)))/( (1/Bi)*(eigV[n-1]/a)*np.sinh(eigV[n-1]/a) + np.cosh(eigV[n-1]/a)))
    
S=np.zeros((201,201))

for i in range(np.size(N,0)):
    S = S + 2*series3[i];


#Sol =  ((1/2)*(1-(x**2)/(L**2)) + (2*series3[0] + 2*series3[1] + 2*series3[2])) * ((q*L**2)/k) + Tinf
    
Sol =  ((1/2)*(1-(x**2)/(L**2)) + (S)) * ((q*L**2)/k) + Tinf

fig1, ax1 = plt.subplots()
cp = ax1.contourf(grid[0],grid[1],Sol)
plt.colorbar(cp)
ax1.set_title('Contour Plot')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.axes.set_aspect('equal')
plt.show()  

#--------------------------------------------------------#
#                   Post Traitement                      #
#--------------------------------------------------------# 

t0 = np.loadtxt(fname = "temperature/Tx0.txt")      # x = 0
t2 = np.loadtxt(fname = "temperature/Tx2.txt")      # x = 0.02
t4 = np.loadtxt(fname = "temperature/Tx4.txt")      # x = 0.04
t6 = np.loadtxt(fname = "temperature/Tx6.txt")      # x = 0.06
t8 = np.loadtxt(fname = "temperature/Tx8.txt")      # x = 0.08
t10 = np.loadtxt(fname = "temperature/Tx10.txt")    # x = 0.1

pl = ['ax2','ax3','ax4','ax6','ax8','ax10']

#for p in pl:
#    ff = plt.figure()
#    ff, p = plt.subplots()
#    p.plot(t0[:,2],t0[:,3])
#    p.set_title('Contour Plot')
#    p.set_xlabel('x (m)')
#    p.set_ylabel('y (m)')
#    plt.show()


fig2, ax2 = plt.subplots()
ax2.plot(grid[1][:,0],Sol[:,0],'-k',label='Solution analytique')
ax2.plot(t0[:,2],t0[:,3],'or',label='Fluent')
ax2.set_title('Evolution de T en x = 0')
ax2.set_xlabel('y (m)')
ax2.set_ylabel('Temperature (K)')
plt.legend()
plt.show()  

fig3, ax3 = plt.subplots()
ax3.plot(grid[1][:,0],Sol[:,40],'-k',label='Solution analytique')
ax3.plot(t2[:,2],t2[:,3],'or',label='Fluent')
ax3.set_title('Evolution de T en x = 0.02')
ax3.set_xlabel('y (m)')
ax3.set_ylabel('Temperature (K)')
plt.legend()
plt.show() 

fig4, ax4 = plt.subplots()
ax4.plot(grid[1][:,0],Sol[:,80],'-k',label='Solution analytique')
ax4.plot(t4[:,2],t4[:,3],'or',label='Fluent')
ax4.set_title('Evolution de T en x = 0.04')
ax4.set_xlabel('y (m)')
ax4.set_ylabel('Temperature (K)')
plt.legend()
plt.show() 

fig5, ax5 = plt.subplots()
ax5.plot(grid[1][:,0],Sol[:,120],'-k',label='Solution analytique')
ax5.plot(t6[:,2],t6[:,3],'or',label='Fluent')
ax5.set_title('Evolution de T en x = 0.06')
ax5.set_xlabel('y (m)')
ax5.set_ylabel('Temperature (K)')
plt.legend()
plt.show() 

fig6, ax6 = plt.subplots()
ax6.plot(grid[1][:,0],Sol[:,160],'-k',label='Solution analytique')
ax6.plot(t8[:,2],t8[:,3],'or',label='Fluent')
ax6.set_title('Evolution de T en x = 0.08')
ax6.set_xlabel('y (m)')
ax6.set_ylabel('Temperature (K)')
plt.legend()
plt.show() 

fig7, ax7 = plt.subplots()
ax7.plot(grid[1][:,0],Sol[:,200],'-k',label='Solution analytique')
ax7.plot(t10[:,2],t10[:,3],':or',label='Fluent')
ax7.set_title('Evolution de T en x = 0.1')
ax7.set_xlabel('y (m)')
ax7.set_ylabel('Temperature (K)')
plt.legend()
plt.show() 

#--------------------------------------------------------#
#                Ordre de Convergence                    #
#--------------------------------------------------------# 
  

xmid_m1 = np.loadtxt(fname = "raff_mesh/xmid_meshm1.txt")
xmid_0 = np.loadtxt(fname = "raff_mesh/xmid_mesh0.txt")
xmid_1 = np.loadtxt(fname = "raff_mesh/xmid_mesh1.txt")
xmid_2 = np.loadtxt(fname = "raff_mesh/xmid_mesh2.txt")
xmid_3 = np.loadtxt(fname = "raff_mesh/xmid_mesh3.txt")


ymid_m1 = np.loadtxt(fname = "raff_mesh/ymid_meshm1.txt")
ymid_0 = np.loadtxt(fname = "raff_mesh/ymid_mesh0.txt")
ymid_1 = np.loadtxt(fname = "raff_mesh/ymid_mesh1.txt")
ymid_2 = np.loadtxt(fname = "raff_mesh/ymid_mesh2.txt")
ymid_3 = np.loadtxt(fname = "raff_mesh/ymid_mesh3.txt")


nxm1 = np.size(ymid_m1,0) - 1
nx0 = np.size(ymid_0,0) 
nx1 = np.size(ymid_1,0) 
nx2 = np.size(ymid_2,0)
nx3 = np.size(ymid_3,0) -1

nym1 = np.size(xmid_m1,0)
ny0 = np.size(xmid_0,0)
ny1 = np.size(xmid_1,0)
ny2 = np.size(xmid_2,0)
ny3 = np.size(xmid_3,0) 

xm1 = np.linspace(0,0.1,nxm1)
x0 = np.linspace(0,0.1,nx0)
x1 = np.linspace(0,0.1,nx1)
x2 = np.linspace(0,0.1,nx2)
x3 = np.linspace(0,0.1,nx3)

ym1 = np.linspace(0,0.05,nym1)
y0 = np.linspace(0,0.05,ny0)
y1 = np.linspace(0,0.05,ny1)
y2 = np.linspace(0,0.05,ny2)
y3 = np.linspace(0,0.05,ny3) 


#--------------------------------------------------------#
#                Ordre de Convergence                    #
#--------------------------------------------------------# 

xp = np.array([xm1,x0,x1,x2,x3]);
yp = np.array([ym1,y0,y1,y2,y3])

seriem1 = [[],[],[],[],[],[],[],[],[],[]]
serie0 = [[],[],[],[],[],[],[],[],[],[]]
serie1 = [[],[],[],[],[],[],[],[],[],[]]
serie2 = [[],[],[],[],[],[],[],[],[],[]]
serie3 = [[],[],[],[],[],[],[],[],[],[]]

gridm1 = np.meshgrid(xm1,ym1)
grid0 = np.meshgrid(x0,y0)
grid1 = np.meshgrid(x1,y1)
grid2 = np.meshgrid(x2,y2)
grid3 = np.meshgrid(x3,y3)

for n in range(1,np.size(N,0)+1):
    seriem1[n-1] = ((-1)**n/eigV[n-1]**3) * ((np.cosh((eigV[n-1]*gridm1[1])/(a*H))*np.cos(eigV[n-1]*(gridm1[0]/L)))/( (1/Bi)*(eigV[n-1]/a)*np.sinh(eigV[n-1]/a) + np.cosh(eigV[n-1]/a)))
    serie0[n-1] = ((-1)**n/eigV[n-1]**3) * ((np.cosh((eigV[n-1]*grid0[1])/(a*H))*np.cos(eigV[n-1]*(grid0[0]/L)))/( (1/Bi)*(eigV[n-1]/a)*np.sinh(eigV[n-1]/a) + np.cosh(eigV[n-1]/a)))
    serie1[n-1] = ((-1)**n/eigV[n-1]**3) * ((np.cosh((eigV[n-1]*grid1[1])/(a*H))*np.cos(eigV[n-1]*(grid1[0]/L)))/( (1/Bi)*(eigV[n-1]/a)*np.sinh(eigV[n-1]/a) + np.cosh(eigV[n-1]/a)))
    serie2[n-1] = ((-1)**n/eigV[n-1]**3) * ((np.cosh((eigV[n-1]*grid2[1])/(a*H))*np.cos(eigV[n-1]*(grid2[0]/L)))/( (1/Bi)*(eigV[n-1]/a)*np.sinh(eigV[n-1]/a) + np.cosh(eigV[n-1]/a)))
    serie3[n-1] = ((-1)**n/eigV[n-1]**3) * ((np.cosh((eigV[n-1]*grid3[1])/(a*H))*np.cos(eigV[n-1]*(grid3[0]/L)))/( (1/Bi)*(eigV[n-1]/a)*np.sinh(eigV[n-1]/a) + np.cosh(eigV[n-1]/a)))

Sm1=np.zeros((nym1,nxm1))
S0=np.zeros((ny0,nx0))
S1=np.zeros((ny1,nx1))
S2=np.zeros((ny2,nx2))
S3=np.zeros((ny3,nx3))

for i in range(np.size(N,0)):
    Sm1 = Sm1 + 2*seriem1[i];
    S0 = S0 + 2*serie0[i];
    S1 = S1 + 2*serie1[i];
    S2 = S2 + 2*serie2[i];
    S3 = S3 + 2*serie3[i];

Solm1 =  ((1/2)*(1-(xm1**2)/(L**2)) + (Sm1)) * ((q*L**2)/k) + Tinf
Sol0 =  ((1/2)*(1-(x0**2)/(L**2)) + (S0)) * ((q*L**2)/k) + Tinf
Sol1 =  ((1/2)*(1-(x1**2)/(L**2)) + (S1)) * ((q*L**2)/k) + Tinf
Sol2 =  ((1/2)*(1-(x2**2)/(L**2)) + (S2)) * ((q*L**2)/k) + Tinf
Sol3 =  ((1/2)*(1-(x3**2)/(L**2)) + (S3)) * ((q*L**2)/k) + Tinf

    
fig8, ax8 = plt.subplots()
cp = ax8.contourf(gridm1[0],gridm1[1],Solm1)
plt.colorbar(cp)
ax8.set_title('Contour Plot')
ax8.set_xlabel('x (m)')
ax8.set_ylabel('y (m)')
ax8.axes.set_aspect('equal')
plt.show()  


#    err = np.divide(np.abs(Gsol[0,1:n+1] - ((Gb-Ga) * (np.exp((rho*u*x2)/Gamma)-1)/( np.exp(rho*u*L/Gamma)-1) + Ga)),G_analitic);
#    err = np.abs(Gsol[0,1:n+1] - G_analitic);
#    norm_err = np.sqrt((1/np.size(Gsol[0,1:n+1]))*np.sum(np.square(err)));
#    norm_err = np.sqrt(np.sum(np.square(err)));
    

#    pp[0,uu] = norm_err;
#        
#    # Ordre de convergence
#    
#for i in range(1,3):
#    ordre[0,i-1]=np.log(pp[0,i-1]/pp[0,i])/np.log(p[i-1]/p[i])

err = np.abs(Solm1[:,int(np.floor(nxm1/2))] - xmid_m1[:,3])
norm_err = np.sqrt((1/np.size(xmid_m1,0))*np.sum(np.square(err)));

err2 = np.abs(Sol0[:,int(np.floor(nx0/2))] - xmid_0[:,3])
norm_err2 = np.sqrt((1/np.size(xmid_0,0))*np.sum(np.square(err2)));

err3 = np.abs(Sol1[:,int(np.floor(nx1/2))] - xmid_1[:,3])
norm_err3 = np.sqrt((1/np.size(xmid_1,0))*np.sum(np.square(err3)));

err4 = np.abs(Sol2[:,int(np.floor(nx2/2))] - xmid_2[:,3])
norm_err4 = np.sqrt((1/np.size(xmid_2,0))*np.sum(np.square(err4)));

err5 = np.abs(Sol3[:,int(np.floor(nx3/2))] - xmid_3[:,3])
norm_err5 = np.sqrt((1/np.size(xmid_3,0))*np.sum(np.square(err5)));


fig9, ax9 = plt.subplots()
ax9.plot(gridm1[1][:,0],Solm1[:,int(np.floor(nxm1/2))],'-k',label='Solution analytique')
ax9.plot(xmid_m1[:,2],xmid_m1[:,3],'or',label='Fluent')
ax9.set_title('Solution en x = 0.05 pour 14 noeuds ')
ax9.set_xlabel('y (m)')
ax9.set_ylabel('Temperature (K)')
plt.legend()
plt.show() 

fig10, ax10 = plt.subplots()
ax10.plot(grid0[1][:,0],Sol0[:,int(np.floor(nx0/2))],'-k',label='Solution analytique')
ax10.plot(xmid_0[:,2],xmid_0[:,3],'or',label='Fluent')
ax10.set_title('Solution en x = 0.05 pour 26 noeuds')
ax10.set_xlabel('y (m)')
ax10.set_ylabel('Temperature (K)')
plt.legend()
plt.show() 

fig11, ax11 = plt.subplots()
ax11.plot(xmid_3[:,2],xmid_3[:,3],'or',label='Fluent')
ax11.plot(grid3[1][:,0],Sol3[:,int(np.floor(nx3/2))],'-k',label='Solution analytique')
ax11.set_title('Solution en x = 0.05 pour 201 noeuds')
ax11.set_xlabel('y (m)')
ax11.set_ylabel('Temperature (K)')
plt.legend()
plt.show() 



#--------------------------------------------------------#
#             Ordre de Convergence (2D)                  #
#--------------------------------------------------------# 

data = np.loadtxt(fname = "temperature/Tmesh1.txt")

xT = data[:,1];
yT = data[:,2];
TT = data[:,3]

serieT = [[],[],[],[],[],[],[],[],[],[]]

for n in range(1,np.size(N,0)+1):
    serieT[n-1] = ((-1)**n/eigV[n-1]**3) * ((np.cosh((eigV[n-1]*yT)/(a*H))*np.cos(eigV[n-1]*(xT/L)))/( (1/Bi)*(eigV[n-1]/a)*np.sinh(eigV[n-1]/a) + np.cosh(eigV[n-1]/a)))

Stest = np.zeros((xT.size,yT.size))

for i in range(np.size(N,0)):
    Stest = Stest + 2*serieT[i];

SolTest =  ((1/2)*(1-(xT**2)/(L**2)) + (Stest)) * ((q*L**2)/k) + Tinf
errT = SolTest[0,:] - TT
norm_errT = np.sqrt((1/np.size(errT,0))*np.sum(np.square(errT)));





data2 = np.loadtxt(fname = "temperature/Tmesh2.txt")

xT2 = data2[:,1];
yT2 = data2[:,2];
TT2 = data2[:,3]

serieT2 = [[],[],[],[],[],[],[],[],[],[]]

for n in range(1,np.size(N,0)+1):
    serieT2[n-1] = ((-1)**n/eigV[n-1]**3) * ((np.cosh((eigV[n-1]*yT2)/(a*H))*np.cos(eigV[n-1]*(xT2/L)))/( (1/Bi)*(eigV[n-1]/a)*np.sinh(eigV[n-1]/a) + np.cosh(eigV[n-1]/a)))

Stest2 = np.zeros((xT2.size,yT2.size))

for i in range(np.size(N,0)):
    Stest2 = Stest2 + 2*serieT2[i];

SolTest2 =  ((1/2)*(1-(xT2**2)/(L**2)) + (Stest2)) * ((q*L**2)/k) + Tinf
errT2 = SolTest2[0,:] - TT2
norm_errT2 = np.sqrt((1/np.size(errT2,0))*np.sum(np.square(errT2)));





data3 = np.loadtxt(fname = "temperature/Tmesh3.txt")

xT3 = data3[:,1];
yT3 = data3[:,2];
TT3 = data3[:,3]

serieT3 = [[],[],[],[],[],[],[],[],[],[]]

for n in range(1,np.size(N,0)+1):
    serieT3[n-1] = ((-1)**n/eigV[n-1]**3) * ((np.cosh((eigV[n-1]*yT3)/(a*H))*np.cos(eigV[n-1]*(xT3/L)))/( (1/Bi)*(eigV[n-1]/a)*np.sinh(eigV[n-1]/a) + np.cosh(eigV[n-1]/a)))

Stest3 = np.zeros((xT3.size,yT3.size))

for i in range(np.size(N,0)):
    Stest3 = Stest3 + 2*serieT3[i];

SolTest3 =  ((1/2)*(1-(xT3**2)/(L**2)) + (Stest3)) * ((q*L**2)/k) + Tinf
errT3 = SolTest3[0,:] - TT3
norm_errT3 = np.sqrt((1/np.size(errT3,0))*np.sum(np.square(errT3)));





data4 = np.loadtxt(fname = "temperature/Tmesh4.txt")

xT4 = data4[:,1];
yT4 = data4[:,2];
TT4 = data4[:,3]

serieT4 = [[],[],[],[],[],[],[],[],[],[]]

for n in range(1,np.size(N,0)+1):
    serieT4[n-1] = ((-1)**n/eigV[n-1]**3) * ((np.cosh((eigV[n-1]*yT4)/(a*H))*np.cos(eigV[n-1]*(xT4/L)))/( (1/Bi)*(eigV[n-1]/a)*np.sinh(eigV[n-1]/a) + np.cosh(eigV[n-1]/a)))

Stest4 = np.zeros((xT4.size,yT4.size))

for i in range(np.size(N,0)):
    Stest4 = Stest4 + 2*serieT4[i];

SolTest4 =  ((1/2)*(1-(xT4**2)/(L**2)) + (Stest4)) * ((q*L**2)/k) + Tinf
errT4 = SolTest4[0,:] - TT4
norm_errT4 = np.sqrt((1/np.size(errT4,0))*np.sum(np.square(errT4)));


op1 = np.log(norm_errT/norm_errT2)/np.log(2)
op2 = np.log(norm_errT2/norm_errT3)/np.log(2)
op3 = np.log(norm_errT3/norm_errT4)/np.log(2)

ordre_moy = (op1+op2+op3)/3


points = np.array([0.01,0.005,0.0025,0.00125])
erreurs = np.array([norm_errT,norm_errT2,norm_errT3,norm_errT4])

#fig12, ax12 =plt.subplots()
#ax12.loglog(points,erreurs,'-o')
#ax12.set_ylabel('$Erreur$')
#ax12.set_xlabel('$\Delta h$')
#ax12 = fig12.add_subplot(111)
#ax12.grid(b=True, which='minor', color='grey', linestyle='--')
#ax12.grid(b=True, which='major', color='k', linestyle='-')

fig=plt.figure()
plt.loglog(points,erreurs,'-ko')
plt.title('Ordre de convergence - TPF1 Cas no. 1 \n Pente = ' + str(round(ordre_moy,3)))
plt.ylabel('$Erreur$')
plt.xlabel('$\Delta h$')
ax = fig.add_subplot(111)
ax.grid(b=True, which='minor', color='grey', linestyle='--')
ax.grid(b=True, which='major', color='k', linestyle='-')
