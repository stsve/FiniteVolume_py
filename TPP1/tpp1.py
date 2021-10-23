import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import mec6616
import numpy as np
import math
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
import time


### init
k=0.5
q=0*1000000
L=0.02
Ta=100
Tb=200

finesseX=20
finesseY=int(finesseX/1)

sparse = True
plot_fin_contour = False
plot_building_fig = False

Precision_par_node = 10**(-1)


def init(finesseXx =finesseX,finesseYy=finesseY):
    global finesseX,finesseY,xx,yy,tri,are,MTri1,fig1, ax1,bcdata
    finesseX=finesseXx
    finesseY=finesseYy

    xx,yy,tri,are = mec6616.RectMesh(0,L,0,L/2,finesseX,finesseY)
    
    #triangulation matplotlib.tri pour graphes
    MTri1 = mtri.Triangulation(xx, yy,tri)   #objet Triangulation pour figures
    
    
    if (plot_building_fig and finesseX+finesseY<30 ):
        #trace la triangulation
        #plt.figure(1)
        fig1, ax1 = plt.subplots()
        ax1.axes.set_aspect('equal')
        plt.triplot(MTri1)
        
        #annoter les noeuds
        for i in range (0,xx.size):
            plt.annotate(str(i),(xx[i],yy[i]))
        #annoter les triangles
        for i in range (0,tri.shape[0]):
            xc = (xx[tri[i,0]]+xx[tri[i,1]]+xx[tri[i,2]])/3.0*1.05
            yc = (yy[tri[i,0]]+yy[tri[i,1]]+yy[tri[i,2]])/3.0 *1.05   
            plt.annotate(str(i),(xc,yc))   
        #annoter les aretes - #numero d'arete-cf
        for i in range (0,are.shape[0]):
            xc = (xx[are[i,0]]+xx[are[i,1]])/2.0
            yc = (yy[are[i,0]]+yy[are[i,1]])/2.0    
            #plt.annotate(str(i)+'-'+str(are[i,4]),(xc,yc)) 
        #plt.show()
        
    #codage des conditions limites - tuple de liste
    bcdata = (['Neumann',Ta],['Neumann',100],['Dirichlet',Tb],['Neumann',200])


### extrapolation aux noeuds

def nodal_interpolation(Sol_aretes):

    A = np.zeros((np.size(xx),np.shape(tri)[0]))
    
    #construction matrice rescensant tous les triangles qui vont impacter un noeud
    for arete in are:
        A[arete[0]][arete[2]]=1
        A[arete[1]][arete[2]]=1
        if (arete[3] != -1):
            A[arete[0]][arete[3]]=1
            A[arete[1]][arete[3]]=1
    
    #ajuste l'impact de chacun de ces triangles en fonction du nbr qu'ils sont à influer chaque noeud
    for a in A:
        a/=sum(a[:])  
    
    #résolution avec la valeur de T du centre des triangles
    
    if (sparse):
        AS = sps.csr_matrix(A)
        Sol = AS.dot(Sol_aretes)
    else:
        Sol = np.matmul(A,Sol_aretes)
    
    #modifie les noeuds limites qui ont des conditions Dirichlet
    for arete in are:
        if (arete[3] == -1 and (bcdata[arete[4]][0] =='Dirichlet')):
            Sol[arete[0]]=bcdata[arete[4]][1]
            Sol[arete[1]]=bcdata[arete[4]][1]
    
    return (Sol)

#### fonctions utiles
def norme_vecteur(a, b):
    return (math.sqrt(math.pow(a,2)+math.pow(b,2)))
    
def centres_triangles(arete):
    num_G = arete[2]
    G=tri[num_G]
    centre_gauche = [(xx[G[0]]+xx[G[1]]+xx[G[2]])/3,(yy[G[0]]+yy[G[1]]+yy[G[2]])/3]
    
    centre_droit = 0
    num_D = arete[3]
    if (num_D!=-1):
        D=tri[num_D]
        centre_droit = [(xx[D[0]]+xx[D[1]]+xx[D[2]])/3,(yy[D[0]]+yy[D[1]]+yy[D[2]])/3] 
    else:
        centre_droit = [(xx[arete[0]]+xx[arete[1]])/2,(yy[arete[0]]+yy[arete[1]])/2]
    return ([centre_droit,centre_gauche])

def vecteur_unitaire(a,b,N):
    return np.array([float(a/N),float(b/N)])


def surface_triangle(triangle):
    a= norme_vecteur((yy[triangle[1]]-yy[triangle[0]]),xx[triangle[1]]-xx[triangle[0]]) #cotés triangle
    b= norme_vecteur((yy[triangle[2]]-yy[triangle[0]]),xx[triangle[2]]-xx[triangle[0]])
    c= norme_vecteur((yy[triangle[2]]-yy[triangle[1]]),xx[triangle[2]]-xx[triangle[1]])
    p=(a+b+c)/2 #demi somme
    return (math.sqrt(p*(p-a)*(p-b)*(p-c)))

def fonction_th (x):
    return(-q/(2*k)*x**2+((Tb-Ta)/L+q/(2*k)*L)*x+Ta)
    
### solveur 2D

def solveur2D(fin_x =finesseX,fin_y=finesseY,plot=True):
    global Sol
    init(finesseXx =fin_x,finesseYy=fin_y)
    seconds = time.time()
    
    ntri = tri.shape[0]
    A = np.zeros((ntri,ntri))
    B_base = np.zeros(ntri)
    for i in range( ntri):
        B_base[i]+=(q*surface_triangle(tri[i]))
    Nnodes = np.size(xx)
    
    initial_nodal = 0
    count=0
    for i in range(4):
        if (bcdata[i][0]=="Dirichlet"):
            initial_nodal+=bcdata[i][1]
            count+=1
    initial_nodal/=count
    
    sol_node = np.ones(Nnodes)*initial_nodal
    
    nb_Are=are.shape[0]
    D=np.zeros(nb_Are)
    S_k=np.zeros(nb_Are)
    
    
    for i in range (nb_Are) :
        arete= are[i]
        
        Ya = yy[arete[1]]-yy[arete[0]] #écart de y entre les 2 pts de l'arrete
        Xa = xx[arete[1]]-xx[arete[0]]
        ETA = norme_vecteur(Ya,Xa)
        
        centres_tri = centres_triangles(arete)
        centre_droit = centres_tri[0]
        centre_gauche = centres_tri[1]
        
        Yc = centre_droit[1]-centre_gauche[1] #écart de y entre les 2 pts centraux
        Xc = centre_droit[0]-centre_gauche[0]
        KSI = norme_vecteur(Yc,Xc)
    
    
        Vect_n=vecteur_unitaire(Ya,-Xa,ETA)
        Vect_eta=vecteur_unitaire(Xa,Ya,ETA)
        Vect_ksi=vecteur_unitaire(Xc,Yc,KSI)
        
        PNKSI = np.dot(Vect_n,Vect_ksi)
        PKSIETA = np.dot(Vect_ksi,Vect_eta)
        
        if (plot_building_fig and finesseX+finesseY<30 ):
            plt.annotate(".",(centre_gauche[0],centre_gauche[1]))
            plt.annotate(".",(centre_droit[0],centre_droit[1]))
            scaleAr=0.1
            plt.quiver((xx[arete[1]]+xx[arete[0]])/2,(yy[arete[1]]+yy[arete[0]])/2,Vect_eta[0]*ETA,Vect_eta[1]*ETA,color = 'red', scale =scaleAr , width = 0.005)
            plt.quiver((xx[arete[1]]+xx[arete[0]])/2,(yy[arete[1]]+yy[arete[0]])/2,Vect_ksi[0]*KSI,Vect_ksi[1]*KSI,color='blue', scale =scaleAr, width = 0.005)
            plt.quiver((xx[arete[1]]+xx[arete[0]])/2,(yy[arete[1]]+yy[arete[0]])/2,Vect_n[0]*ETA,Vect_n[1]*ETA ,scale =scaleAr,width = 0.005)
    
    
        D[i]= k/PNKSI*ETA/KSI
    
        #   Aretes internes 
        if (not(arete[3]==-1 and bcdata[arete[4]][0]=='Neumann')):
            A[arete[2],arete[2]]+=D[i]
        if (arete[3]!=-1):
            A[arete[3],arete[3]]+=D[i]
            A[arete[2],arete[3]]-=D[i]
            A[arete[3],arete[2]]-=D[i]
        
        #   Aretes en frontière : Membre de droite
        if (arete[3]==-1 and bcdata[arete[4]][0] == 'Dirichlet' ) :
            B_base[arete[2]]+=(bcdata[arete[4]][1]*D[i])
        S_k[i]=-k*PKSIETA/PNKSI
    print("pre_construction time : "+str(round(time.time()-seconds,2))+" seconds\nStart iterating on nodes")
    ecart_nodal=[]
    flag_stop=False
    flag_div=False
    iter =0
    
    while ( not (flag_stop or iter>30)):
        B = np.copy(B_base)
        for i in range(nb_Are) :
            
            arete=are[i]
            
            S=S_k[i]*(sol_node[arete[1]]-sol_node[arete[0]])
            
            #   Aretes internes : Membre de droite
            B[arete[2]]+=S
            if (arete[3]!=-1):
                B[arete[3]]-=S 
        
        if (sparse):
            AS = sps.csr_matrix(A)
            Sol = spsolve(AS,B)
        else : 
            Sol = np.linalg.solve(A, B)
        
        sol_node_temp = nodal_interpolation(Sol)
        ecart_nodal.append(np.sum(abs(sol_node-sol_node_temp)))
        sol_node = sol_node_temp
        
        if (iter>=1):
            if (ecart_nodal[iter]>ecart_nodal[iter-1]):
                flag_div=True
                print("test_divergence")
            else:
                if (flag_div):
                    flag_div=False
                    print("ok")
                if (ecart_nodal[iter]/Nnodes<Precision_par_node):
                    print("précision atteinte")
                    flag_stop=True
                
        iter+=1
        # if (iter==15):
        #     flag_stop=True
        
    
    # print (A)
    # print(B)
    print ("sparse : "+str(sparse))
    print (str(iter)+" itérations")
    pTime= round(time.time()-seconds,2)
    print("processing time : "+str(pTime)+" seconds")
    
    if (plot):
        fig3, ax3 = plt.subplots()
        plt.title(str(iter)+" iterations\nprécision : "+str(Precision_par_node)+"\nprocessing time : "+str(pTime)+" seconds")
        if (plot_fin_contour):
            tcf = ax3.tricontourf(MTri1, sol_node)
            ax3.tricontour(MTri1, sol_node, colors='k')
            ax3.axes.set_aspect("equal")
            plt.triplot(MTri1,color='red',lw=0.05)
        else:
            tcf = ax3.tripcolor(MTri1, facecolors=Sol, edgecolors='k')
            ax3.axes.set_aspect("equal")
            plt.triplot(MTri1,color='black',lw=0.1)
        
        fig3.colorbar(tcf)
        
        #annoter les noeuds
        if (finesseX+finesseY<20 ):
            for i in range (0,xx.size):
                plt.annotate(str(round(sol_node[i],2)),(xx[i],yy[i]),color='red') 
        
        plt.show()
    else:
        return pTime
        

### sparse

def print_sparse_influence(plot=True):
    global sparse,Y_s,Y_ns
    plt.title("sparse influence")
    X= [10,20,30,35,40,45,50,55]
    
    # #attention, pour avoir un résultat fiable il faut contraindre le nombre d'itération ( à 15 par ex)
    # Y_s=np.zeros(len(X))
    # Y_ns=np.zeros(len(X))
    # for i in range (len(X)):
    #     print("\n\n____________  "+str(round((i/len(X))**2*100))+"% sparse influence _________ \n\n")
    #     sparse=True
    #     Y_s[i]=solveur2D(X[i],X[i],plot=False)
    #     sparse=False
    #     Y_ns[i]=solveur2D(X[i],X[i],plot=False)
    
    Y_s=[ 0.09,  0.75,  2.97,  5.16,  9.11, 14.44, 21.18, 31. ]
    Y_ns=[ 0.23,  1.05,  3.74,  6.92, 12.66, 20.87, 33.22, 51.32]
    
    plt.plot(X,Y_s,label='sparse')
    plt.plot(X,Y_ns,label='no_sparse')
    plt.xlabel("finesse maillage")
    plt.ylabel("temps de calcul")
    plt.legend()
    if (plot):
        plt.show()
    
### erreur
def erreur(plot=True):
    
    analytique=np.zeros(len(Sol))
    for i in range(len(Sol)):
        analytique[i]=fonction_th((xx[tri[i][0]]+xx[tri[i][1]]+xx[tri[i][2]])/3)
    if (plot):
        fig5, ax5 = plt.subplots()
        ax5.axes.set_aspect("equal")
        tcf = ax5.tripcolor(MTri1, facecolors=analytique, edgecolors='k')
        plt.triplot(MTri1,color='black',lw=0.1)
        fig5.colorbar(tcf)
        plt.title("solution analytique")
        plt.show()
    
    
    Erreur=np.zeros(len(Sol))
    for i in range(len(Sol)):
        Erreur[i]=Sol[i]-fonction_th((xx[tri[i][0]]+xx[tri[i][1]]+xx[tri[i][2]])/3)
    if (plot):
        fig4, ax4 = plt.subplots()
        ax4.axes.set_aspect("equal")
        tcf = ax4.tripcolor(MTri1, facecolors=Erreur, edgecolors='k')
        plt.triplot(MTri1,color='black',lw=0.1)
        fig4.colorbar(tcf)
        plt.title("Erreur aux triangles")
        plt.show()
    return(math.sqrt(np.sum(np.square(Erreur)))/len(Sol))

### convergeance

def ordre_de_conv(plot=True):
        
    #X= np.array([2,4,8,16,32,64])
    X= np.array([3,6,12,24,48])
    
    nb=len(X)
    x= X[:nb-1] 
    N_conv =np.zeros(nb-1)
    solveur2D(X[0],X[0],plot=False)
    valPrec=erreur(plot=False)
    for i in range (1,nb):
        print("\n\n____________  "+str(round((i/nb)**4*100))+"% ordre de convergence _________ \n\n")
        solveur2D(X[i],X[i],plot=False)
        valTemp=erreur(plot=False)
        N_conv[i-1]=math.log(valPrec/valTemp)/math.log(X[i]/X[i-1])
        valPrec=valTemp
    
    # x= np.array([4,8,16,32,64])
    # N_conv=np.array([2.34145281 , 2.70370949 , 3.09207745 , 1.09051045 , 0.0162877 ])
    
    plt.figure("ordre de convergence")
    plt.title("moyenne : "+ str(round(np.sum(abs(N_conv))/len(N_conv),2)))
    plt.plot(x,N_conv,linestyle='-',marker='o',label="ordre de convergence")
    plt.legend()
    plt.xscale('log', basex=2)
    print(N_conv)
    if (plot):
        plt.show()

### solution manufacturés

def MMS(x):
    T0 = -100
    T1 = 190
    T2 = 180
    a0 = 0.1
    a1 = 10
    a2 = 0.8
    L = 0.02
    return (T0 + T1*np.cos(a0*math.pi*x/L) + T2*np.sin(a2*math.pi*x/L))

def plot_mms(plot=True):
    mms=np.zeros(len(Sol))
    for i in range(len(Sol)):
        mms[i]=MMS((xx[tri[i][0]]+xx[tri[i][1]]+xx[tri[i][2]])/3)
    if (plot):
        fig5, ax5 = plt.subplots()
        ax5.axes.set_aspect("equal")
        tcf = ax5.tripcolor(MTri1, facecolors=mms, edgecolors='k')
        plt.triplot(MTri1,color='black',lw=0.1)
        fig5.colorbar(tcf)
        plt.title("MMS")
        plt.show()

###

    
solveur2D()
#print_sparse_influence()
print("erreur par point : "+str(erreur()))
#ordre_de_conv()
#plot_mms()


