import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io 
import matplotlib.tri as mtri
import matplotlib.tri.triangulation as tri 
from mpl_toolkits.mplot3d import Axes3D



#%% PARAMÈTRES DÉTERMINANTS LE CALCUL
################################################

maille =  1  # Choix du maillage 1 (gors),2 (plus fin) ,3 (encore plus fin) 

# Proprietes mecaniques
E = 1000 # Young's modulus  
v = 0.3  # Poisson's coefficient

# Forces (force linéique sur le bord x=2 ici)
fsx = 100         
fsy = 0
fs = np.array([fsx,fsy])



#%% LECTURE ET CHARGEMENT DU MAILAGE
#######################################################

if maille == 1:
    mat = scipy.io.loadmat('maillage_1.mat')
elif maille ==2:
    mat = scipy.io.loadmat('maillage_2.mat')
elif maille ==3:
    mat = scipy.io.loadmat('maillage_3.mat')
else:
    raise ValueError('Maille 1,2, ou 3')
    
# paramètres déduits
#===============================================

# coordonnes des noeuds  
X = mat['p'].T  

# table de connectivite
C = mat['t'].T[:,:-1] -1   

# nombre de noeuds du maillage
n_nodes = X.shape[0] 

# nombre d'elements du maillage
n_elems = C.shape[0] 

# Nombre de noeuds par element
n_nodes_elem = 3   

# Nombre de ddl par element
ndofe= 2*n_nodes_elem

# Nombre de ddl dans tout le domaine  
ndof = 2*n_nodes 


# dimensions de la plaque 
xmin =  np.min(X[:,0])
xmax =  np.max(X[:,0])
ymin =  np.min(X[:,1])
ymax =  np.max(X[:,1])
# vérifier dimensions de la plaque [0,2]x[0,1]

# affichage du maillage
Initial_triangulation =  tri.Triangulation(X[:,0],X[:,1],C)
plt.figure(0)
plt.triplot(Initial_triangulation, color = 'black')
plt.title("maillage")
plt.show()


#%% CALCUL DES FUNS EF P1 LAGRANGE
############################################
 
def fun_tri_P1_lag(x,y,x_nodes,y_nodes):
    # x,y -> point d'évaluation
    # x_nodes -> tableau 1D numpy avec abscisses des noeuds
    # y_nodes -> tableau 1D numpy avec ordonnées des noeuds
    
    x1=x_nodes[0]; x2=x_nodes[1]; x3=x_nodes[2];
    y1=y_nodes[0]; y2=y_nodes[1]; y3=y_nodes[2];
    
    Te =0.5*np.abs((x2-x1)*(y3-y1)-(y2-y1)*(x3-x1))
    
    N1 = (x2*y3-x3*y2+(y2-y3)*x+(x3-x2)*y)/(2*Te)
    N2 = (x3*y1-x1*y3+(y3-y1)*x+(x1-x3)*y)/(2*Te)
    N3 = (x1*y2-x2*y1+(y1-y2)*x+(x2-x1)*y)/(2*Te)
    
    dN1dx = (y2-y3)/(2*Te)
    dN1dy = (x3-x2)/(2*Te)
    dN2dx = (y3-y1)/(2*Te)
    dN2dy = (x1-x3)/(2*Te)
    dN3dx = (y1-y2)/(2*Te)
    dN3dy = (x2-x1)/(2*Te)
    
    N  = np.array([N1,N2,N3])
    dNdx = np.array([dN1dx,dN2dx,dN3dx])
    dNdy = np.array([dN1dy,dN2dy,dN3dy])
    
    return [N,dNdx,dNdy]
    # N = [N1(x,y),N2(..),N3(..)]-> tableau 1D numpy avec valeur des 3 funs au point d'éval
    # dNdx -> idem avec dérivée par rapport à x des funs
    # dNdy -> idem avec dérivée par rapport à y des funs
    
def GetBe(dNdx,dNdy):
    # dNdx=[dN1dx(x,y),dN2dx(x,y),dN3dx(x,y)] -> tableau 1D numpy avec valeur à un certain point d'éval
    row1 = np.r_[dNdx,np.zeros(n_nodes_elem)]
    row2 = np.r_[np.zeros(n_nodes_elem),dNdy]
    row3 = np.r_[dNdy,dNdx]
    Be    = np.c_[row1,row2,row3].T
    return Be
    # Be : matrice élémentaire lien déformation - déplacement (evaluee à un certain point)


def GetNe(N):
    row1 = np.r_[N,np.zeros(n_nodes_elem)]
    row2 = np.r_[np.zeros(n_nodes_elem),N] 
    Ne_matrix = np.c_[row1,row2].T
    return Ne_matrix
    # Ne_matrix : matrice élémentaire lien déplacement - ddls 

#%%
#--------------------------------------------------------------------------
#
# 1. CALCUL DES DEPLACEMENTS
#
#--------------------------------------------------------------------------
    

#%% CALCUL DE LA MATRICE DE RIGIDITE
#######################################################
    

# Loi de comportement : Contraintes - Deformations
# Hypothese de contraintes planes 
# convention : [sigmaxx,sigmayy,sigmaxy] = H [epsxx, epsyy, 2*epsxy] 
H = E/(1-v**2)*np.array([[1,v,0],
                         [v,1,0],
                         [0,0,0.5*(1-v)]])

    
# initialisation de la matrice de rigiditié 
K = np.zeros((ndof,ndof))

# Boucle sur les éléments

for e in range(n_elems):
    
    # coordonnes des noeuds de l'element e   
    x_nodes = X[C[e,:],0]
    y_nodes = X[C[e,:],1] 
    
    x1=x_nodes[0]; x2=x_nodes[1]; x3=x_nodes[2] 
    y1=y_nodes[0]; y2=y_nodes[1]; y3=y_nodes[2] 
    
    # surface de l'élément e
    Te = 0.5*np.abs((x2-x1)*(y3-y1)-(y2-y1)*(x3-x1))

    # calcul de la matrice de rigidité elementaire
    [Nm,dNmdx,dNmdy] = fun_tri_P1_lag(x1,y1,x_nodes,y_nodes)
    Be = GetBe(dNmdx,dNmdy)
    Ke = Te*(Be.T.dot(H.dot(Be)))

     
    # Assemblage des contributions élémentaires
    for i in range(n_nodes_elem):
        for j in range(n_nodes_elem):
            K[C[e,i],C[e,j]] += Ke[i,j]
            K[C[e,i],C[e,j]+n_nodes]  += Ke[i,j+n_nodes_elem]
            K[C[e,i]+n_nodes,C[e,j]]  += Ke[i+n_nodes_elem,j]
            K[C[e,i]+n_nodes,C[e,j]+n_nodes]  += Ke[i+n_nodes_elem,j+n_nodes_elem]
        

#%% CALCUL DU SECOND MEMBRE
#######################################################
           
# initialisation du second membre
F = np.zeros(ndof)

# Boucle sur les noeuds recevant la force

nd = 0 # compteur pour compter les noeuds sur le bord droit

for n in range(n_nodes):# boucle sur l'ensemble des noeuds
    
    if X[n,0] == xmax:      # arete recevant la force 
        
        nd = nd+1

        F[n] = fsx
        F[n+n_nodes] = fsy
        
        if X[n,1] == ymin:   # cas du premier coin (bas)
            F[n] = fsx/2
            F[n+n_nodes] = fsy/2
        
        if X[n,1] == ymax:    # cas du second coin (haut)
            F[n] = fsx/2
            F[n+n_nodes] = fsy/2


F = F*(ymax-ymin)/(nd-1)
        
 

#%% IMPOSITION DES CL DE DIRICHLET
#######################################################
        
        
for n in range(n_nodes):# boucle sur les noeuds
    
    # depl Ux=0 en x=0
    if X[n,0] == xmin: # noeuds bloqués
        
        K[n,:] = 0 
        K[:,n] = 0 
        K[n,n]=1
        
        F[n]=0
    
    # depl Uy=0 en y=0    
    elif X[n,1] == ymin:
        
        K[n+n_nodes,:] = 0 
        K[:,n+ n_nodes] =0
        K[n+n_nodes, n+n_nodes]=1
        
        F[n+ n_nodes] = 0

    
#%% RESOLUTION DU SYSTÈME LINÉAIRE ET VISUALISATION
#######################################################
        
# résolution
U = np.linalg.solve(K,F)

#Calcul des coordonnes des noeuds apres deformation
x = X[:,0] + U[:n_nodes]
y = X[:,1] + U[n_nodes:]

# affichage du maillage
Deformed_triangulation  =  tri.Triangulation(x,y,C)
plt.figure(1)
plt.triplot(Initial_triangulation, color = 'black')
plt.triplot(Deformed_triangulation, color = 'blue')
plt.show()
 

#%%
#--------------------------------------------------------------------------
#
# 2. POST-TRAITEMENT
#
#--------------------------------------------------------------------------
    

#%% TRACÉ DE LA SOLUTION UX LE LONG DU COTE Y=0
####################################################

UUx = []# liste des Ux du bord inférieur
XXx = []# liste des x du bord inférieur

## A VOUS DE JOUER ICI !!!
    
plt.figure(2)
plt.plot(XXx,UUx,'-or')
plt.ylabel('Ux')
plt.xlabel('x')
plt.title('Ux=f(x)')
plt.grid()
plt.show()



#%% CALCUL DE LA CONTRAINTE SIGXX
##################################
# initialisations

T = np.zeros(n_nodes)       # Surface
SIGXX = np.zeros(n_nodes)     # contrainte


#boucle sur les elements 

for e in range(n_elems):
    
    # coordonnes des noeuds de l'element e   
    x_nodes = X[C[e,:],0]
    y_nodes = X[C[e,:],1] 
    
    x1=x_nodes[0]; x2=x_nodes[1]; x3=x_nodes[2] 
    y1=y_nodes[0]; y2=y_nodes[1]; y3=y_nodes[2] 
    
    # surface de l'élément e
    Te = 0.5*np.abs((x2-x1)*(y3-y1)-(y2-y1)*(x3-x1))
    
    # Calcul des contraintes de Von Mises
    [Nm,dNmdx,dNmdy] = fun_tri_P1_lag(x1,y1,x_nodes,y_nodes)
    Be = GetBe(dNmdx,dNmdy)
    Ue = np.r_[U[C[e,:]],U[C[e,:]+n_nodes]]
    sigma = H.dot(Be.dot(Ue))
    sigxx = sigma[0]
    
    # Assemblage : surface et sigma
    for i in range(n_nodes_elem):
        T[C[e,i]] +=  Te
        SIGXX[C[e,i]] +=  Te*sigxx

# Calcul de la contrainte de Von Mises aux noeuds
SIGXX = SIGXX/T

# Visualisation des containtes 
t = mtri.Triangulation(x,y,C)
plt.figure(3)
l = np.array([np.min(SIGXX),np.max(SIGXX)+10])
l =  np.linspace(l[0],l[1], 20)
plt.triplot(t)
plt.tricontourf(t,SIGXX,levels=l)
plt.colorbar()
plt.title('Contraintes sig_xx')



#%% CALCUL DE LA DEFO EPSXX
##################################


#%% CALCUL DE LA DEFO EPSXX
##################################


T = np.zeros(n_nodes)       # Surface
EPSXX = np.zeros(n_nodes)   # defo epsxx


#boucle sur les elements 

## A VOUS DE JOUER ICI !!!

# Visualisation de la defo
t = mtri.Triangulation(x,y,C)
plt.figure(4)
l = np.array([np.min(EPSXX),np.max(EPSXX)+1e-1])
l =  np.linspace(l[0],l[1], 20)
plt.triplot(t)
plt.tricontourf(t,EPSXX,levels=l)
plt.colorbar()
plt.title('Defo epsxx')

 
 
 
    



 
