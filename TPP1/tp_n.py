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
finesseY=finesseX

sparse =True
plot_fin=True
plot_fin_contour = True
trace_fig = True

Precision_par_node = 10**(-8)


def init(finesseXx =finesseX,finesseYy=finesseY):
    global finesseX,finesseY,xx,yy,tri,are,MTri1,fig1, ax1,bcdata, centri
    finesseX=finesseXx
    finesseY=finesseYy

    xx,yy,tri,are = mec6616.RectMesh(0,L,0,L/2,finesseX,finesseY)
    
    #triangulation matplotlib.tri pour graphes
    MTri1 = mtri.Triangulation(xx, yy,tri)   #objet Triangulation pour figures
    
    
    centri = np.zeros((np.size(tri,0),2))

    u = 0;
    for t in tri:
        centri[u,0] = np.sum(xx[t])/3
        centri[u,1] = np.sum(yy[t])/3
        u += 1
    
    if (trace_fig and finesseX+finesseY<30 ):
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
    bcdata = (['Neumann',0],['Neumann',0],['Dirichlet',Tb],['Neumann',0])



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

def solveur2D(fin_x =finesseX,fin_y=finesseY):
    global Sol
    init(finesseXx =fin_x,finesseYy=fin_y)
    seconds = time.time()
    
    ntri = tri.shape[0]
    A = np.zeros((ntri,ntri))
    B_base = np.zeros(ntri)
    
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
        
        if (trace_fig and finesseX+finesseY<30 ):
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
        B_base[arete[2]]+=q*1/2*surface_triangle(tri[arete[2]])
        S_k[i]=-k*PKSIETA/PNKSI
#        print(surface_triangle(tri[arete[2]]))
    print("pre_construction time : "+str(round(time.time()-seconds,2))+" seconds\nStart iterating on nodes")
    
    ecart_nodal=[]
    flag_stop=False
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
                #flag_stop=True
                print("stop par divergence")
            elif (ecart_nodal[iter]/Nnodes<Precision_par_node):
                print("précision atteinte")
                flag_stop=True
        iter+=1
        # if (iter==10):
        #     flag_stop=True
        
    
    print (A)
    print(B)
    print ("sparse : "+str(sparse))
    print (str(iter)+" itérations")
    pTime= round(time.time()-seconds,2)
    print("processing time : "+str(pTime)+" seconds")
    
    if (plot_fin):
        fig3, ax3 = plt.subplots()
        plt.title(str(iter)+" iterations\nprécision : "+str(Precision_par_node)+"\nprocessing time : "+str(pTime)+" seconds")
        if (plot_fin_contour):
            tcf = ax3.tricontourf(MTri1, sol_node)
            ax3.tricontour(MTri1, sol_node, colors='k')
            plt.triplot(MTri1,color='red',lw=0.05)
        else:
            tcf = ax3.tripcolor(MTri1, facecolors=Sol, edgecolors='k')
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

def print_sparse_influence():
    global sparse,plot_fin,Y_s,Y_ns
    plot_fin=False
    plt.title("sparse influence")
    X= [10,20,30,35,40,45,50,55]
    
    # Y_s=np.zeros(len(X))
    # Y_ns=np.zeros(len(X))
    # for i in range (len(X)):
    #     sparse=True
    #     Y_s[i]=solveur2D(X[i],X[i])
    #     sparse=False
    #     Y_ns[i]=solveur2D(X[i],X[i])
    
    Y_s=[ 0.09,  0.75,  2.97,  5.16,  9.11, 14.44, 21.18, 31. ]
    Y_ns=[ 0.23,  1.05,  3.74,  6.92, 12.66, 20.87, 33.22, 51.32]
    
    plot_fin=True
    plt.plot(X,Y_s,label='sparse')
    plt.plot(X,Y_ns,label='no_sparse')
    plt.xlabel("finesse maillage")
    plt.ylabel("temps de calcul")
    plt.legend()
    plt.show()
    
### erreur
def erreur():
#    e=0
    for i in range(len(Sol)):
        Sol[i]-=fonction_th((xx[tri[i][0]]+xx[tri[i][1]]+xx[tri[i][2]])/3)
    fig4, ax4 = plt.subplots()
    tcf = ax4.tripcolor(MTri1, facecolors=Sol, edgecolors='k')
    plt.triplot(MTri1,color='black',lw=0.1)
    fig4.colorbar(tcf)
    plt.show()

###   
    
solveur2D()
#print_sparse_influence()
#erreur()
plt.show()


#-----------------------------------------------------------------------------#
#                               Post Traitement                               #
#-----------------------------------------------------------------------------# 

X = np.unique(centri[:,0])
Y = np.unique(centri[:,1])

XX =  np.zeros((np.size(X),finesseX)) 
YY =  np.zeros((np.size(Y),finesseY))

u=0;
for i in range(0,np.size(X)):
    XX[u,:]=X[u]
    YY[u,:]=Y[u]
    u+=1

T_ana = ( ((Tb-Ta)/L)+q/(2*k) * (L-XX) )*XX + Ta;

#
#SolanaXY = np.zeros((np.size(Y),np.size(X)))
#
#for i in range(0,np.size(Y)):
#    SolanaXY[i,:] = T_ana
#    
#grid = np.meshgrid(X,Y)
#
#fig4, ax4 = plt.subplots()
#cp = ax4.contourf(grid[0],grid[1],SolanaXY,levels = 250)
#plt.colorbar(cp)
#ax4.set_title('Contour Plot')
#ax4.set_xlabel('x (m)')
#ax4.set_ylabel('y (m)')
#ax4.axes.set_aspect('equal')
#plt.show()  
