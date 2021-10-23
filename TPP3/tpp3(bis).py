import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import mec6616
import gmsh
import numpy as np
import math
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
import sympy as sp
import random
import sys

import time


### init

X0=0
Xf=10
Y0=0
Yf=1


Source=np.array([0,0])
rho=1

upwind = True
build_plot = False
plot_leased_squares = False
plot_fin_contour = False
plot_building_fig = False

x,y,u,v = sp.symbols('x y u v')

Precision_par_tri = 10**(-5)
plot_inter=50

H = 1
poiseuille = 1.5 * (4*y/H-4*y**2/H**2) 
# u = poiseuille
# Fu = sp.lambdify([x,y],u,'numpy')

u=1
Fu = u

v=0
Fv = v#sp.lambdify([x,y],v,'numpy')

P = 0
# P = 2*y+x**2
Fp = P#sp.lambdify([x,y], P,'numpy')
Fpn = 0#sp.lambdify([x,y], (sp.diff(P,y)),'numpy')

Re=100
# nu=0.01
nu = 1*Yf*rho/Re

##

def geometrie(bool, finesseXx=12,finesseYy=6):
    global finesseX, finesseY, lc, plot_inter, xx, yy, tri, are, MTri1, ETA, KSI, PNKSI, PKSIETA, Vect_n, Vect_eta, Vect_ksi, Ct, Ca, St, plot_building_fig
    finesseX=finesseXx
    finesseY=finesseYy
    if (bool): 
        xx,yy,tri,are = mec6616.RectMesh(X0,Xf,Y0,Yf,finesseX,finesseY)
        lc = (Xf-X0)/finesseX
    else :
        lc = 0.2 
        xx,yy,tri,are = mec6616.RectGmsh(X0,Xf,Y0,Yf,lc)

# =============================================================================
#     H1 = 0.5
#     H2 = 1
#     L1 = 16
#     L2 = 4
#     xx,yy,tri,are = mec6616.BackstepGmsh(H1,H2,L1,L2,lc)
# =============================================================================
    
    MTri1 = mtri.Triangulation(xx, yy,tri) 

    nTri = tri.shape[0]
    nAre=are.shape[0]
    plot_inter = min (max (round(30000/nTri),10),150)

    PNKSI = np.zeros(nAre)
    PKSIETA = np.zeros(nAre)
        
    Ct = centre_triangle(tri)
    St = surface_triangle(tri)
    Ca = centre_arete(are)
    
    Ya = yy[are[:,1]]-yy[are[:,0]] #écart de y entre les 2 pts de l'arrete
    Xa = xx[are[:,1]]-xx[are[:,0]]
    ETA = norme_vecteur(Ya,Xa)
    
    Vect_n = np.transpose(vecteur_unitaire(Ya,-Xa,ETA))
    Vect_eta = np.transpose(vecteur_unitaire(Xa,Ya,ETA))

    centre_droit = centres_dt()
    centre_gauche = Ct[are[:,2]]
    
    Yc = centre_droit[:,1]-centre_gauche[:,1] #écart de y entre les 2 pts centraux
    Xc = centre_droit[:,0]-centre_gauche[:,0]
    KSI = norme_vecteur(Yc,Xc)
    Vect_ksi = np.transpose(vecteur_unitaire(Xc,Yc,KSI))
    
    for i in range (nAre) :
        PNKSI[i] = np.dot(Vect_n[i],Vect_ksi[i])
        PKSIETA[i] = np.dot(Vect_ksi[i],Vect_eta[i])
    
    
    if (plot_building_fig and finesseX+finesseY<=30 ):
        #trace la triangulation
        fig1, ax1 = plt.subplots()
        # ax1.axes.set_aspect('equal')
        plt.triplot(MTri1, color = "k" , lw= "0.1")
        
        #annoter les noeuds
        for i in range (xx.size):
            plt.annotate(str(i),(xx[i],yy[i]))
        #annoter les triangles
        for i in range (tri.shape[0]):
            xc =Ct[i,0]*1.05
            yc = Ct[i,1] *1.05   
            plt.annotate(str(i),(xc,yc))  
            plt.scatter(Ct[i,0],Ct[i,1],color = "k",s=0.2)
        #annoter les aretes - #numero d'arete-cf
        for i in range (are.shape[0]):
            arete  = are[i]
            #plt.annotate(str(i)+'-'+str(are[i,4]),(Ca[i,0],Ca[i,1])) 
            plt.scatter(Ca[i,0],Ca[i,1],color = "red",s=0.2)
            if (arete[3]==-1):
                plt.scatter(centre_droit[i][0],centre_droit[i][1],color = "k",s=0.2)
        plt.quiver(Ca[:,0],Ca[:,1],Vect_eta[:,0]*ETA[:],Vect_eta[:,1]*ETA[:],color = 'red',  width = 0.005)
        plt.quiver(Ca[:,0],Ca[:,1],Vect_ksi[:,0]*KSI[:],Vect_ksi[:,1]*KSI[:],color='blue',  width = 0.005)
        plt.quiver(Ca[:,0],Ca[:,1],Vect_n[:,0]*ETA[:],Vect_n[:,1]*ETA[:] ,width = 0.005)
        
    
def CL():
    global bcdata, plot_fluid_vel
    
    bcdata_u = (['Dirichlet',Fu],['Dirichlet',0],['Neumann',0],['Dirichlet',0])
    bcdata_v = (['Dirichlet',0],['Dirichlet',0],['Neumann',0],['Dirichlet',0])
    # bcdata_p = (['Dirichlet',Fp],['Neumann',Fpn] ,['Libre',0],['Neumann',Fpn])
    bcdata_p = (['Libre',Fp],['Neumann',Fpn],['Dirichlet',Fp],['Neumann',Fpn])
    bcdata_Pgeo = ['Entree','Paroi','Sortie','Paroi']

    bcdata = [bcdata_u, bcdata_v, bcdata_p, bcdata_Pgeo]
    
    
### solveur 2D

def solveur2D (bool ,fin_x ,fin_y):
    global SolX, SolY, sol_nodeX, sol_nodeY, P_tri, AX, AY, convergence_plot, RESIDUS_plot, DIVERGENCE_plot, build_plot
    geometrie(bool,finesseXx =fin_x,finesseYy=fin_y)
    CL()
    seconds = time.time()
    convergence_plot, (RESIDUS_plot, DIVERGENCE_plot) = plt.subplots(1,2,sharex=True)
    convergence_plot.canvas.set_window_title("Convergence") 
    RESIDUS_plot.set_yscale('log', basey=10)
    RESIDUS_plot.set_title("Résidus")
    DIVERGENCE_plot.set_yscale('log', basey=10)
    DIVERGENCE_plot.set_title("Divergence")
    plt.pause(0.0001)


    ntri = tri.shape[0]
    print(str(ntri) + " triangles")
    nb_Are = are.shape[0]

    A_baseX = np.zeros((ntri, ntri))
    B_baseX = np.zeros(ntri)
    
    A_baseY = np.zeros((ntri, ntri))
    B_baseY =  np.zeros(ntri)

    sol_nodeX = np.zeros(np.size(xx))
    sol_nodeY = np.zeros(np.size(xx))
    SolX = Sol_tempX=np.zeros(ntri)
    SolY = Sol_tempY=np.zeros(ntri)
    
    init_LS(bcdata[2])
    init_nodal()
    
    Uf = Uf_temp = np.zeros(nb_Are)
    Grad = np.zeros ((ntri ,2))
    P_tri = np.zeros(ntri)
    div = []
    
    S_k = - nu * PKSIETA / PNKSI
    
    for i in range (nb_Are) :
        
        Di = nu/PNKSI[i]*ETA[i]/KSI[i]
        basesD(i, A_baseX, B_baseX, bcdata[0], Di)
        basesD(i, A_baseY, B_baseY, bcdata[1], Di)
    
    ecart_incremental=[]
    flag_stop=False
    flag_div=False
    iter =0
    
    print("Start iterating ("+str(round(time.time()-seconds,2))+ " sec)")
    for iter in range (300):
    # while ( not flag_stop and iter<500):            
        AX , B_baseX2 = np.copy(A_baseX) , np.copy(B_baseX)
        AY , B_baseY2 = np.copy(A_baseY) , np.copy(B_baseY)
        
        B_baseX2 -= Grad[:,0]*St
        B_baseY2 -= Grad[:,1]*St

        for i in range (nb_Are) :
                      
            Fi = Uf[i] * rho * ETA[i]
            basesF(i, AX, B_baseX2, bcdata[0], Fi)
            basesF(i, AY, B_baseY2, bcdata[1], Fi)
        
        
        for sous_iter in range(2):
            BX, BY  = np.copy(B_baseX2), np.copy(B_baseY2)
            for i in range(nb_Are) :
                arete=are[i]
                SX , SY = S_k[i]*(sol_nodeX[arete[1]]-sol_nodeX[arete[0]]) , S_k[i]*(sol_nodeY[arete[1]]-sol_nodeY[arete[0]])
                B_fin(i, BX, SX)
                B_fin(i, BY, SY)
                
            SolX , SolY = spsolve(sps.csr_matrix(AX),BX) , spsolve(sps.csr_matrix(AY),BY)
            sol_nodeX , sol_nodeY = nodal_interpolation(SolX, SolY)


        Uf_RC = Rhie_Chow (Grad, False)
        Uf = Relaxation(Uf_RC,Uf)
        div.append(np.sum(np.abs(div_champ(Uf))))
        Pp = Correcteur_pression(Uf, bcdata[3] )
        # print("%0.1e"%np.sum(np.abs(div_champ(Uf))))
        P_tri = RelaxationP(P_tri, Pp)
        least_squares(bcdata[2], Grad)
        
        if (iter>0):
            ecart_incremental.append(np.sum(np.sqrt(np.square(SolX-Sol_tempX)+np.square(SolY-Sol_tempY))))
            flag_div, flag_stop = divergence_test(flag_div, ecart_incremental, div, iter-1, ntri)
        
        Sol_tempX, Sol_tempY = SolX, SolY
        
            
        iter+=1
        
    pTime= round(time.time()-seconds,2)
    tt=((str(round(pTime//60))+":"+("0" if pTime%60<10 else "")) if pTime//60>0 else "")+str(round(pTime%60,2))+ (" min" if pTime//60>0 else " sec")
    
    print (("upwind, " if upwind else "centré, ")+"Re = "+(str(1*Yf/(nu/rho))+"\n")+str(iter)+" iterations en "+tt+"\nprécision : "+str("%0.1e"%(ecart_incremental[-1]/ntri)))
    
    
    EI = np.asarray(ecart_incremental)/ntri
    RESIDUS_plot.scatter(np.arange(len(EI))+2,EI)
    DIVERGENCE_plot.scatter(np.arange(len(div))+1,div)
    

### extrapolation aux noeuds

def init_nodal():
    global AS
    A = np.zeros((np.size(xx),np.shape(tri)[0]))
    line_sum = np.zeros(np.size(xx))
    #construction matrice rescensant tous les triangles qui vont impacter un noeud
    for arete in are:
        if (A[arete[0]][arete[2]]==0):
            line_sum[arete[0]] +=1
        if (A[arete[1]][arete[2]]==0):
            line_sum[arete[1]] +=1
            
        A[arete[0]][arete[2]]=1
        A[arete[1]][arete[2]]=1
        
        if (arete[3] != -1):
            if (A[arete[0]][arete[3]]==0):
                line_sum[arete[0]] +=1
            if (A[arete[1]][arete[3]]==0):
                line_sum[arete[1]] +=1
                
            A[arete[0]][arete[3]]=1
            A[arete[1]][arete[3]]=1
    
    #ajuste l'impact de chacun de ces triangles en fonction du nbr qu'ils sont à influer chaque noeud
    for i in range (len(A)):
        A[i,:]/=line_sum[i]
    AS = sps.csr_matrix(A)
    

def nodal_interpolation(Sol_tri_X , Sol_tri_Y):
    
    Sol_nod_X, Sol_nod_Y = AS.dot(Sol_tri_X), AS.dot(Sol_tri_Y)
    
    #modifie les noeuds limites qui ont des conditions Dirichlet
    correcteur_nodal(Sol_nod_X,bcdata[0])
    correcteur_nodal(Sol_nod_Y,bcdata[1])
    
    return (Sol_nod_X,Sol_nod_Y)

###Rhie Chow

def Rhie_Chow ( Grad, plot=True ):
    DAU = 1/2 * (np.diag(AX)+np.diag(AY))
    # least_squares(bcdata[2],Grad)
    nb_are = are.shape[0]
    Uff= np.zeros(nb_are)

    for i in range( nb_are):
        arete = are[i]
        Vp_ap = St[arete[2]]/DAU[arete[2]]
        Va_aa = St[arete[3]]/DAU[arete[3]]
        if (arete[3] !=-1):
            moyV = np.dot(np.array([SolX[arete[2]],SolY[arete[2]]])+np.array([SolX[arete[3]],SolY[arete[3]]]),Vect_n[i])
            pres = (Vp_ap+Va_aa)*(P_tri[arete[2]]-P_tri[arete[3]])/KSI[i]
            p_grad = np.dot(Vp_ap*Grad[arete[2]]+Va_aa*Grad[arete[3]],Vect_ksi[i])
            Uff[i]= 1/2 *(moyV + pres + p_grad)
        elif(bcdata[0][arete[4]][0] == 'Dirichlet'and bcdata[1][arete[4]][0] == 'Dirichlet' ):
            Uff[i]=np.dot(np.array([data_value(bcdata[0][arete[4]][1],Ca[i][0],Ca[i][1]),data_value(bcdata[1][arete[4]][1],Ca[i][0],Ca[i][1])]),Vect_n[i])
        elif (bcdata[0][arete[4]][0] == 'Neumann' and bcdata[1][arete[4]][0] == 'Neumann'):
            moyV = np.dot(np.array([SolX[arete[2]],SolY[arete[2]]]),Vect_n[i])
            press = 0
            if (bcdata[2][arete[4]][0] == 'Dirichlet'):
                pres = Vp_ap*(P_tri[arete[2]]-data_value(bcdata[2][arete[4]][1],Ca[i][0],Ca[i][1]))/KSI[i]
            p_grad = np.dot(Vp_ap*Grad[arete[2]],Vect_ksi[i])
            Uff[i]= (moyV + pres + p_grad)


    if (plot):
        fig1b, ax1b = plt.subplots()
        # ax1b.axes.set_aspect('equal')
        plt.triplot(MTri1,color ="k", lw = 0.1)        
        plt.title ("extrapolation de vitesse aux aretes")
        ax1b.quiver(np.concatenate((Ca[:,0], Ct[:,0])),np.concatenate((Ca[:,1], Ct[:,1])),np.concatenate((Vect_n[:,0]*Uff, SolX)),np.concatenate((Vect_n[:,1]*Uff, SolY)),width=0.003,color = 'black')
    
    return Uff


##Correction pression

def Correcteur_pression(Uf , bcGeo):
    ntri = tri.shape[0]
    nb_are = are.shape[0]
    
    Mp = np.zeros((ntri, ntri))
    b = np.zeros(ntri)
    DAU = 1/2 * (np.diag(AX)+np.diag(AY))
    df = np.zeros(nb_are)
    
    for i in range( nb_are):
        arete = are[i]
        df[i] = (St[arete[2]]/DAU[arete[2]] + St[arete[3]]/DAU[arete[3]])/(2*KSI[i])
        d = rho * ETA[i]
        Dd = d * df[i]
        
        b[arete[2]] -= d * Uf[i]
        
        if (arete[3] != -1 ) :
            Mp[arete[2],arete[2]] += Dd
            Mp[arete[3],arete[3]] += Dd
            Mp[arete[3],arete[2]] -= Dd
            Mp[arete[2],arete[3]] -= Dd
            
            b[arete[3]] += d * Uf[i]
            
        elif (bcGeo[arete[4]] == 'Sortie'):
            df[i] = (St[arete[2]]/DAU[arete[2]])/KSI[i]
            Mp[arete[2],arete[2]] += d * df[i]
    
    P_p = spsolve(sps.csr_matrix(Mp),b) 
    
    for i in range( nb_are):
        arete = are[i]
        if (arete[3] != -1 ) :
            Uf[i] += df[i]*(P_p[arete[2]] - P_p[arete[3]])
        elif(bcGeo[arete[4]] == 'Sortie'):
            Uf[i] += df[i]*P_p[arete[2]]
    
    return P_p
           


### Least_squares
def init_LS (data):
    global ATA
    ntri = tri.shape[0]
    nb_Are = are.shape[0]

    ATA = np.zeros((ntri,2,2))
    AP = np.zeros((2,2))
    
    for i in range (nb_Are) :
        arete = are[i]
        if (arete[3] !=-1 or data[arete[4]][0] != 'Libre' ) :
            V = KSI[i] * Vect_ksi[i]
            AP[0,0] = V[0]**2
            AP[0,1] = AP[1,0] = V[0]*V[1]
            AP[1,1] = V[1]**2
            
            if (arete[3] ==-1 and data[arete[4]][0] == 'Neumann' ) :
                
                AP[0,0] *= Vect_n[i,0]**2 
                AP[0,1] *= Vect_n[i,0]*Vect_n[i,1]
                AP[1,0] = AP[0,1]
                AP[1,1] *= Vect_n[i,1]**2
            
            ATA [arete[2]] += AP
            
            if (arete[3] !=-1):
                ATA [arete[3]] += AP
                

def least_squares (data, Grad, plot=False):
    global plot_leased_squares
    ntri = tri.shape[0]
    nb_Are = are.shape[0]

    B = np.zeros ((ntri ,2))
    
    for i in range (nb_Are) :
        arete = are[i]
        
        if (arete[3] !=-1):
            
            b= KSI[i]*(P_tri[arete[3]]-P_tri[arete[2]])
            B[arete[2]] += b*Vect_ksi[i]
            B[arete[3]] += b*Vect_ksi[i]
            
        elif (data[arete[4]][0] == 'Dirichlet' ):
            b= KSI[i]*(data_value(data[arete[4]][1],Ca[i,0],Ca[i,1])-P_tri[arete[2]])
            B[arete[2]] += b*Vect_ksi[i]
            
        elif  (data[arete[4]][0] == 'Neumann' ):
            a = P_tri[arete[2]] - data_value(data[arete[4]][1],Ct[arete[2],0],Ca[i,0]) * KSI[i] * ETA[i] * PKSIETA[i]
            b = KSI[i] * ( a - P_tri[arete[2]])
            B[arete[2]] += b * Vect_ksi[i]
    
    for i in range (ntri) :
        Grad[i] = np.linalg.solve(ATA[i], B[i])
    
    
    if(plot_leased_squares and finesseX+finesseY<50):
        # plot_leased_squares=False
        fig1b, ax1b = plt.subplots()
        # ax1b.axes.set_aspect('equal')
        plt.triplot(MTri1,color ="k", lw = 0.1)
          
        Fc = np.zeros(ntri)
        for i in range (ntri):
           Fc[i]=math.sqrt(Grad[i][0]**2+Grad[i][1]**2)
        tcf=ax1b.tripcolor(MTri1, facecolors=Fc)
        fig1b.colorbar(tcf)
        plt.title ("Gradient de pression")
        ax1b.quiver(Ct[:,0],Ct[:,1],Grad[:,0],Grad[:,1],width=0.003,color = 'red')
        if (plot):
            plt.show()


def conv_LS(plot = False):
    X= np.array([3,4,6,8,12,16,24,32,48,64])#,96,128])
    nb=len(X)
    E=np.zeros(nb)
    for i in range (nb):
        E[i] =mms_LS(X[i],X[i])

    
    Ordre = np.polyfit(np.log((Xf-X0)/X), np.log(E), 1)
    regress = np.poly1d(Ordre)
    pente =round(Ordre[0],2)
    
    OrdreIni = np.polyfit(np.log((Xf-X0)/X[:3]), np.log(E[:3]), 1)
    regressIni = np.poly1d(OrdreIni)
    penteIni =round(OrdreIni[0],2)
    x=[]
    for i in range(nb):
        print("conv "+str(i))
        a= regressIni(np.log((Xf-X0)/X[i]))
        if (a>=np.log(E[-1]) or a>= regress(np.log((Xf-X0)/X[-1]))):
            x.append(X[i])
    x=np.array(x)
    
    plt.figure("Ordre de convergence "+("upwind " if upwind else "centré ") )
    plt.title("Ordre de convergence "+("upwind " if upwind else "centré ") )
    plt.loglog((Xf-X0)/X,E,linestyle='-',marker='o',label="Erreur",color='k')
    plt.loglog((Xf-X0)/X,np.exp(regress(np.log((Xf-X0)/X))),label='Regression linéaire = ' + str(pente))
    plt.loglog((Xf-X0)/x,np.exp(regressIni(np.log((Xf-X0)/x))),label='Regression linéaire ini = ' + str(penteIni))
    plt.ylabel('$Erreur$')
    plt.xlabel('$\Delta h$')
    plt.legend()
    plt.grid(b=True, which='minor', color='grey', linestyle='--')
    plt.grid(b=True, which='major', color='k', linestyle='-')
        
    if (plot):
        plt.show()
    return (penteIni)
    
def mms_LS(divX,divY,build_plot=True):
    geometrie(divX,divY)
    CL()
    least_squares(bcdata[2])
    
    gx = sp.lambdify([x,y],sp.simplify(sp.diff(P,x)),'numpy')
    gy = sp.lambdify([x,y],sp.simplify(sp.diff(P,y)),'numpy')
    ntri = tri.shape[0]
    analytique=np.zeros((ntri,2))
    # an=np.zeros(ntri)
    for i in range(ntri):
        analytique[i]=np.array([gx(Ct[i,0],Ct[i,1]),gy(Ct[i,0],Ct[i,1])])
    #     an[i] = norme_vecteur(gx(Ct[i,0],Ct[i,1]),gy(Ct[i,0],Ct[i,1]))
    # 
    # if (build_plot):
    #     fig5, ax5 = plt.subplots()
    #     ax5.axes.set_aspect("equal")
    #     tcf = ax5.tripcolor(MTri1, facecolors=an, edgecolors='k')
    #     for i in range (ntri):
    #         ax5.quiver(Ct[i,0],Ct[i,1],analytique[i,0],analytique[i,1],scale = 50,width=0.003,color = 'red')
    #     plt.triplot(MTri1,color='black',lw=0.1)
    #     fig5.colorbar(tcf)
        # plt.title("solution analytique")
        # if (plot):
        #     plt.show()
    
    Erreur=np.zeros(ntri)
    for i in range(ntri):
        Erreur[i]=norme_vecteur((Grad[i,0]-analytique[i,0]),(Grad[i,1]-analytique[i,1]))
        
    # if (build_plot):
    #     fig4, ax4 = plt.subplots()
    #     ax4.axes.set_aspect("equal")
    #     tcf = ax4.tripcolor(MTri1, facecolors=Erreur, edgecolors='k')
    #     plt.triplot(MTri1,color='black',lw=0.1)
    #     fig4.colorbar(tcf)
        # plt.title("Erreur aux triangles ")
        # plt.show()
            
    return (math.sqrt(np.sum(np.square(Erreur))/ntri))
    


#### fonctions utiles
def norme_vecteur(a, b):
    return ((a**2+b**2)**(1/2))
    
    
def centre_triangle(triangles):
    return(np.transpose(np.array([(xx[triangles[:,0]]+xx[triangles[:,1]]+xx[triangles[:,2]])/3,(yy[triangles[:,0]]+yy[triangles[:,1]]+yy[triangles[:,2]])/3])))


def centre_arete(aretes):
    return(np.transpose(np.array([(xx[aretes[:,0]]+xx[aretes[:,1]])/2,(yy[aretes[:,0]]+yy[aretes[:,1]])/2])))


def centres_dt():
    N = are.shape[0]
    Cd = np.zeros((N,2))
    for N_arete in range( N):
        num_D = are[N_arete,3]
        if (num_D!=-1):
            Cd [N_arete] = Ct[num_D]
        else:
            Cd [N_arete] = Ca[N_arete]
    return Cd

def vecteur_unitaire(a,b,N):
    return np.array([a/N,b/N])


def surface_triangle(triangles):
    a= norme_vecteur((yy[triangles[:,1]]-yy[triangles[:,0]]),xx[triangles[:,1]]-xx[triangles[:,0]]) #cotés triangles
    b= norme_vecteur((yy[triangles[:,2]]-yy[triangles[:,0]]),xx[triangles[:,2]]-xx[triangles[:,0]])
    c= norme_vecteur((yy[triangles[:,2]]-yy[triangles[:,1]]),xx[triangles[:,2]]-xx[triangles[:,1]])
    p=(a+b+c)/2 #demi somme
    return ((p*(p-a)*(p-b)*(p-c))**(1/2))
    
def data_value(data, x, y):
    return (data if isinstance(data, (int, float)) else data(x,y))
    
    
def correcteur_nodal (Sol, data):
    # l'object Sol se fait modifer instantanement
    for arete in are:
        if (arete[3] == -1 and (data[arete[4]][0] =='Dirichlet')):
            Sol[arete[0]] = data_value(data[arete[4]][1],xx[arete[0]],yy[arete[0]])
            # Sol[arete[1]] = data_value(data[arete[4]][1],xx[arete[1]],yy[arete[1]])


def basesD (i, A, B , data, Di):
    arete = are[i]
    # les objects A et B se font modifer instantanement
    if (arete[3]!=-1):
        A[arete[2],arete[2]]+= Di
        A[arete[3],arete[3]]+= Di
        A[arete[2],arete[3]]-= Di
        A[arete[3],arete[2]]-= Di
    
    if (arete[3]==-1 and data[arete[4]][0] == 'Dirichlet' ) :
            A[arete[2],arete[2]] += Di
            B[arete[2]] += data_value(data[arete[4]][1],Ca[i,0],Ca[i,1])*Di


def basesF (i, A, B , data, Fi):
    arete = are[i]
    # les objects A et B se font modifer instantanement
    if (arete[3]!=-1):
        A[arete[2],arete[2]]+= ((+Fi/2) if not upwind else (+max(Fi,0)))
        A[arete[3],arete[3]]+= ((-Fi/2) if not upwind else (+max(0,-Fi)))
        A[arete[2],arete[3]]+= ((+Fi/2) if not upwind else (-max(0,-Fi)))
        A[arete[3],arete[2]]+= ((-Fi/2) if not upwind else (-max(Fi,0)))
    
    if (arete[3]==-1):
        if (data[arete[4]][0]=='Neumann'):
            A[arete[2],arete[2]]+= Fi
        
        elif (data[arete[4]][0] == 'Dirichlet' ) :
            A[arete[2],arete[2]]+= max(0,Fi)
            B[arete[2]]+= data_value(data[arete[4]][1],Ca[i,0],Ca[i,1])*max(0,-Fi)


def B_fin (i, B, S):
    arete = are[i]
    B[arete[2]]+=S
    if (arete[3]!=-1):
        B[arete[3]]-=S 


def Relaxation (Sol_1, Sol_2):
    relax = 0.1
    return relax*Sol_1 +(1-relax)*Sol_2
    
def RelaxationP (Sol_1, Sol_2):
    relax = 0.1
    return Sol_1 + relax*Sol_2

def divergence_test(flag_div, ecart_incremental, div, iter, ntri):
    global build_plot, convergence_plot, RESIDUS_plot, DIVERGENCE_plot
    flag_stop = False
    if (iter //plot_inter!=0 and iter%plot_inter ==0):
        EI = np.asarray(ecart_incremental)/ntri
        # if (plt.isinteractive()) : 
        plt.close(convergence_plot)

        convergence_plot, (RESIDUS_plot, DIVERGENCE_plot) = plt.subplots(1,2,sharex=True)
        convergence_plot.canvas.set_window_title("Convergence") 
        RESIDUS_plot.set_yscale('log', basey=10)
        RESIDUS_plot.set_title("Résidus")
        DIVERGENCE_plot.set_yscale('log', basey=10)
        DIVERGENCE_plot.set_title("Divergence")
        # else:
        #     del RESIDUS_plot.collections[:]
        #     del DIVERGENCE_plot.collections[:]
        RESIDUS_plot.scatter(np.arange(len(EI))+2,EI)
        DIVERGENCE_plot.scatter(np.arange(len(div))+1,div)
        # build_plot =True
        plt.pause(0.0001)
        
        
    
    if (iter>2 and ecart_incremental[iter]>ecart_incremental[iter-1]):
        if (not flag_div):
            # sys.stdout.write("\r")
            # print("La solution est en train de diverger ")
        # else:
            flag_div=True
            # sys.stdout.write("\r")
            # print("test_divergence ")
    else:
        if (flag_div):
            flag_div=False
            # sys.stdout.write("\r")
            # print("ok ")
            
        if (ecart_incremental[iter]/ntri<Precision_par_tri):
            # print("précision atteinte ("+ str("%.0e"%Precision_par_tri)+")")
            flag_stop=True
    return flag_div, flag_stop

### divergeance champs constant

def div_champ(Uf):
    
    divTri = np.zeros(tri.shape[0])
    for i in range(are.shape[0]):
        arete = are[i]
        c = ETA[i] * Uf[i]         
        divTri[arete[2]] -= c
        if (arete[3] !=-1 ):
            divTri[arete[3]] += c
    
    return (divTri)
    

###coupes

def coupe(Xx=[],Yy=[],theo =False,plot=True , Vx=True , Vy =False):
    axe = ("y" if (len(Xx)==0) else "x")
    plt.figure("coupe "+axe+(" upwind " if upwind else " centré ") )
    plt.title("coupe "+axe+(" upwind " if upwind else " centré ")+"\nVitesse " +("" if (Vx and Vy) else ("U" if (Vx) else "V")))
       
    for y in Yy:
        X,V,diff = coupe_uni(y , 1 , Vx ,Vy)
        X,V,diff = elimination_lointains(diff, X, V)
        X,V = extrapolation_par_plan (diff, X, V)            
        
        plt.plot(X,V,linestyle='-',marker='o',label="y = "+str(y))
        
        plt.legend()
        
    for x in Xx :
        Y,V,diff = coupe_uni(x , 0 , Vx ,Vy)
        Y,V,diff = elimination_lointains(diff, Y, V)
        Y,V = extrapolation_par_plan (diff, Y, V)

        plt.plot(Y,V,linestyle='-',marker='o',label="x = "+str(x))
        if (theo):
            H = (1 if round(Y[-1]-Y[0])==1 else 0.5)
            if (Y[0]<0):
                y=Y+0.5
                c=1/2
            else :
                y=Y
                c=1
            plt.plot( Y,c*1.5 * (4*y/H-4*y**2/H**2))
      
        plt.legend()
      
    if (plot):    
        plt.show()

def coupe_uni (x , XOY, Vx, Vy):
    coor= np.where(abs(Ct[:,XOY]-x)<lc)[0]
    
    V = np.sqrt(np.square(SolX[coor] )+np.square(SolY[coor])) if (Vx and Vy) else (SolX[coor] if (Vx) else SolY[coor])
    diff = Ct[coor,XOY]-x
    Y = Ct[coor,(not XOY)*1]
    
    coorNode = np.where(abs((yy if(XOY) else xx)-x)<lc)[0]
    Vn = np.sqrt(np.square(sol_nodeX[coorNode] )+np.square(sol_nodeY[coorNode])) if (Vx and Vy) else (sol_nodeX[coorNode] if (Vx) else sol_nodeY[coorNode])
    V = np.concatenate((V , Vn))
    diffN = (yy[coorNode] if(XOY) else xx[coorNode])-x
    diff = np.concatenate((diff, diffN))
    Yn = (xx[coorNode] if(XOY) else yy[coorNode])
    Y = np.concatenate((Y, Yn))
    
    indxs = Y.argsort()
    y = Y[indxs]
    v = V[indxs]
    diff = diff[indxs]
    return (y,v,diff)
    
def elimination_lointains(diff, y, v):
    dim = len(y)
    pos = 0
    indx = []
    dd = []
    for i in range(dim):
        if (len(dd)==0 and diff[i]==0):
            indx.append(i)
            pos=0
        elif (pos==0 or pos * np.sign(diff[i])==1 ):
            dd.append(diff[i])
            pos = np.sign(diff[i])
        else:
            M = pos*min(np.abs(dd))
            ind = dd.index(M)
            indx.append(i-len(dd)+ind)
            # dd.remove(M)
            # if (len(dd)!=0):
            #     M = pos*min(np.abs(dd))
            #     inde = dd.index(M)
            #     indx.append(i-(1+len(dd))+(inde+1 if inde>=ind else inde))
            dd=[diff[i]]
            pos*=-1
            
    diff = diff[indx]
    v = v[indx]
    y = y[indx]
    return (y,v,diff)
    
def extrapolation_par_plan(diff, y, v):
    
    dim = len(diff)
    Y = []
    V = []
        
    V.append(v[0])
    Y.append(y[0])

    for i in range(1,dim-1):
        AB = [y[i-1]-y[i],v[i-1]-v[i]]
        AC = [y[i-1]-y[i+1],v[i-1]-v[i+1]]
        
        B= [diff[i]-diff[i-1],diff[i+1]-diff[i-1]]
        m = (np.asarray([AB,AC]))
        if(np.linalg.det(m)!=0):
            plan = np.linalg.solve(m, 1*np.asarray(B))
            d = -(1*diff[i]+plan[0]*y[i]+plan[1]*v[i])
            Y.append(np.average(y[i-1:i+2]))
            V.append(-(plan[0]*Y[-1]+d )/ plan[1])
   
    V.append(v[dim-1])
    Y.append(y[dim-1])
    
    return (np.asarray(Y),np.asarray(V))

### convergeance

def ordre_de_conv(plot=True):
        
    X= np.array([3,4,6,8,12,16,24,32,48,64])#,96,128])
    
    nb=len(X)
    E=np.zeros(nb)
    E[0] =MMS(init_MMS() , X[0],X[0],build_plot =False)
    for i in range (1,nb):
        print("\n\n____________  "+str(round((i/nb)**4*100))+"% ordre de convergence _________ \n\n")
        print("Calcul de finesse : "+str(X[i]))
        E[i] =MMS(Source,X[i],X[i],build_plot =False)

    
    Ordre = np.polyfit(np.log((Xf-X0)/X), np.log(E), 1)
    regress = np.poly1d(Ordre)
    pente =round(Ordre[0],2)
    
    OrdreIni = np.polyfit(np.log((Xf-X0)/X[:3]), np.log(E[:3]), 1)
    regressIni = np.poly1d(OrdreIni)
    penteIni =round(OrdreIni[0],2)
    x=[]
    for i in range(nb):
        a= regressIni(np.log((Xf-X0)/X[i]))
        if (a>=np.log(E[-1]) or a>= regress(np.log((Xf-X0)/X[-1]))):
            x.append(X[i])
    x=np.array(x)
    
    plt.figure("Ordre de convergence "+("upwind " if upwind else "centré ") )
    plt.title("Ordre de convergence "+("upwind " if upwind else "centré ") )
    plt.loglog((Xf-X0)/X,E,linestyle='-',marker='o',label="Erreur",color='k')
    plt.loglog((Xf-X0)/X,np.exp(regress(np.log((Xf-X0)/X))),label='Regression linéaire = ' + str(pente))
    plt.loglog((Xf-X0)/x,np.exp(regressIni(np.log((Xf-X0)/x))),label='Regression linéaire ini = ' + str(penteIni))
    plt.ylabel('$Erreur$')
    plt.xlabel('$\Delta h$')
    plt.legend()
    plt.grid(b=True, which='minor', color='grey', linestyle='--')
    plt.grid(b=True, which='major', color='k', linestyle='-')
        
    if (plot):
        plt.show()
    return (penteIni)

###célik

def ordre_par_célik(plot=True):
        
    X= np.array([4,8,16])#,32])#,64])
    #X= np.array([3,6,12,24,48])#,96])

    
    nb=len(X)
    x= X[:nb-2] 
    N_conv =np.zeros(nb-2)
    E=np.zeros(nb)
    E[0] =MMS(init_MMS(), X[0],X[0],build_plot=False)
    for i in range(1,nb):
        print("\n\n____________  "+str(round((i/nb)**4*100))+"% ordre de célik _________ \n\n")
        E[i] =MMS(Source,X[i],X[i],build_plot=False)
    for i in range (nb-2):
        N_conv[i]=math.log(abs((E[i+1]-E[i]))/abs((E[i+2]-E[i+1])))/math.log(math.sqrt(2))
    
    moy =round(np.average(N_conv),2)
    plt.figure("ordre de célik , moyenne = " +str(moy))
    plt.title("ordre de célik "+("upwind " if upwind else "centré ")+"\n    moyenne = " +str(moy))
    plt.plot(x,N_conv,linestyle='-',marker='o',label="ordre de convergence")
    plt.legend()
    plt.xscale('log', basex=2)
    if (plot):
        plt.show()
    return (moy)
    


### solution manufacturés
def init_MMS():
    global Fp,Fpn,P_tri
    diffu = sp.simplify(sp.diff(sp.diff(u,x),x)+sp.diff(sp.diff(u,y),y))
    convu= sp.simplify(sp.diff(u*u*rho,x)+sp.diff(v*u*rho,y))
    sourceu = convu - nu* diffu
    
    diffv = sp.simplify(sp.diff(sp.diff(v,x),x)+sp.diff(sp.diff(v,y),y))
    convv= sp.simplify(sp.diff(v*v*rho,y)+sp.diff(v*u*rho,x))
    sourcev = convv - nu* diffv
    
    P = sp.sqrt(sp.Pow(sourceu,2)+sp.Pow(sourcev,2))
    Fp = sp.lambdify([x,y], P,'numpy')
    Fpn = sp.lambdify([x,y],sp.diff(P,y),'numpy')

    P_tri = Fp (Ct[:,0],Ct[:,1])
    if (str(P_tri).isdigit()):
        P_tri = np.ones(tri.shape[0])*Fp

    
    return np.array([sp.lambdify([x,y],sourceu,'numpy'),sp.lambdify([x,y],sourcev,'numpy')])
    

def MMS(source, finesseXx, finesseYy, build_plot=True,plot=False):
    global Source
    Source = source
    
    solveur2D(finesseXx,finesseYy,build_plot,plot)
    
    nTri = tri.shape[0]
    analytique=np.zeros((nTri,2))
    an=np.zeros(nTri)
    for i in range(nTri):
        analytique[i]=np.array([Fu(Ct[i,0],Ct[i,1]),Fv(Ct[i,0],Ct[i,1])])
        an[i] = norme_vecteur(Fu(Ct[i,0],Ct[i,1]),Fv(Ct[i,0],Ct[i,1]))
    if (build_plot):
        fig5, ax5 = plt.subplots()
        ax5.axes.set_aspect("equal")
        tcf = ax5.tripcolor(MTri1, facecolors=an, edgecolors='k')
        ax5.quiver(Ct[:,0],Ct[:,1],analytique[:,0],analytique[:,1],width=0.003,color = 'red')
        plt.triplot(MTri1,color='black',lw=0.1)
        fig5.colorbar(tcf)
        plt.title("solution analytique")
        if (plot):
            plt.show()
    
    
    Erreur=np.zeros(nTri)
    for i in range(nTri):
        Erreur[i]=norme_vecteur((SolX[i]-Fu(Ct[i,0],Ct[i,1])),(SolY[i]-Fv(Ct[i,0],Ct[i,1])))
        
    if (build_plot):
        fig4, ax4 = plt.subplots()
        ax4.axes.set_aspect("equal")
        tcf = ax4.tripcolor(MTri1, facecolors=Erreur, edgecolors='k')
        plt.triplot(MTri1,color='black',lw=0.1)
        fig4.colorbar(tcf)
        plt.title("Erreur aux triangles "+("upwind " if upwind else "centré "))
        if (plot):
            plt.show()
    Ec = math.sqrt(np.sum(np.square(Erreur))/nTri)
    print("Erreur/noeud : "+str("%.2e"%Ec))
    return(Ec)
    
###

#divX=10
#divY=int(divX/2)
#solveur2D(True ,divX,divY)
#coupe(Xx=[0.5, 3, 4.5, 9],theo =True, plot=False)
#plt.show()
#
#divX=20
#divY=int(divX/2)
#solveur2D(True , divX,divY)
#coupe(Xx=[0.5, 3, 4.5, 9],theo =True, plot=False)
#plt.show()

divX=40
divY=int(divX/2)
solveur2D(True , divX,divY)
coupe(Xx=[0.5, 3, 4.5, 9],theo =True, plot=False)
plt.show()
#
#upwind=False
#solveur2D(False , divX,divY)
#coupe(Xx=[0.5, 3, 4.5, 9],theo =True, plot=False)
#plt.show()
