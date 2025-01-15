import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io 
import matplotlib.tri as mtri
import matplotlib.tri.triangulation as tri 
from mpl_toolkits.mplot3d import Axes3D



#%% PARAMÈTRES DÉTERMINANTS LE CALCUL
################################################

maille1 = 1  # Choix du maillage 1 (gors),2 (plus fin) ,3 (encore plus fin)

# Proprietes mecaniques
E1 = 1000 ; E2 = 2000 # Young's modulus  
v1 = 0. ; v2 = 0.  # Poisson's coefficient

# Forces (force linéique sur le bord x=2 ici)
fsx = 100         
fsy = 0
fs = np.array([fsx,fsy])



#%% LECTURE ET CHARGEMENT DU MAILAGE
#######################################################

if maille1 == 1:
    mat = scipy.io.loadmat('maillage_1.mat')
elif maille1 ==2:
    mat = scipy.io.loadmat('maillage_2.mat')
elif maille1 ==3:
    mat = scipy.io.loadmat('maillage_3.mat')
else:
    raise ValueError('Maille 1,2 ou 3')
    
# paramètres déduits
#===============================================

# coordonnes des noeuds  
X1 = mat['p'].T
X2 = np.copy(X1)
X2[:,0] = X1[:,0]+2
  

# table de connectivite
C1 = mat['t'].T[:,:-1] -1
C2 = mat['t'].T[:,:-1] -1     

# nombre de noeuds du maillage
n_nodes1 = X1.shape[0]
n_nodes2 = X2.shape[0] 

# nombre d'elements du maillage
n_elems1 = C1.shape[0]
n_elems2 = C2.shape[0] 

# Nombre de noeuds par element
n_nodes_elem = 3   

# Nombre de ddl par element
ndofe= 2*n_nodes_elem

# Nombre de ddl dans tout le domaine  
ndof1 = 2*n_nodes1
ndof2 = 2*n_nodes2 


# dimensions de la plaque 
xmin1 =  np.min(X1[:,0])
xmax1 =  np.max(X1[:,0])
ymin1 =  np.min(X1[:,1])
ymax1 =  np.max(X1[:,1])
xmin2 =  np.min(X2[:,0])
xmax2 =  np.max(X2[:,0])
ymin2 =  np.min(X2[:,1])
ymax2 =  np.max(X2[:,1])


# affichage du maillage
Initial_triangulation1 =  tri.Triangulation(X1[:,0],X1[:,1],C1)
Initial_triangulation2 =  tri.Triangulation(X2[:,0],X2[:,1],C2)
plt.figure(0)
plt.triplot(Initial_triangulation1, color = 'black')
plt.triplot(Initial_triangulation2, color = 'blue')
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
H1 = E1/(1-v1**2)*np.array([[1,v1,0],
                         [v1,1,0],
                         [0,0,0.5*(1-v1)]])
H2 = E2/(1-v2**2)*np.array([[1,v2,0],
                         [v2,1,0],
                         [0,0,0.5*(1-v2)]])

    
# initialisation de la matrice de rigiditié 
K1 = np.zeros((ndof1,ndof1))
K2 = np.zeros((ndof2,ndof2))

## A VOUS DE JOUER ICI !!!

        

#%% CALCUL DES OPERATEURS MORTAR
#######################################################

## A VOUS DE JOUER ICI !!!           

    
            
#%% CALCUL DU SECOND MEMBRE
#######################################################
           
# initialisation du second membre
F1 = np.zeros(ndof1)
F2 = np.zeros(ndof2)

## A VOUS DE JOUER ICI !!!
        
 

#%% IMPOSITION DES CL DE DIRICHLET
#######################################################
        
## A VOUS DE JOUER ICI !!!        


    
#%% RESOLUTION DU SYSTÈME LINÉAIRE ET VISUALISATION
#######################################################
        
# Construction du système augmenté puis résolution
        
## A VOUS DE JOUER ICI !!!


#Calcul des coordonnes des noeuds apres deformation
xxx1 = X1[:,0] + U1[:n_nodes1]
yyy1 = X1[:,1] + U1[n_nodes1:]

xxx2 = X2[:,0] + U2[:n_nodes2]
yyy2 = X2[:,1] + U2[n_nodes2:]

# affichage du maillage
Deformed_triangulation1  =  tri.Triangulation(xxx1,yyy1,C1)
Deformed_triangulation2  =  tri.Triangulation(xxx2,yyy2,C2)
plt.figure(1)
plt.triplot(Deformed_triangulation1, color = 'black')
plt.triplot(Deformed_triangulation2, color = 'blue')
plt.show()
 

#%%
#--------------------------------------------------------------------------
#
# 2. POST-TRAITEMENT : CALCUL DES CONTRAINTES DE VON MISES
#
#--------------------------------------------------------------------------
    

#%% TRACÉ DE LA SOLUTION UX LE LONG DU COTE Y=0
####################################################

UUx = []
XXx = []

## A VOUS DE JOUER ICI !!!
        
    
plt.figure(2)
plt.plot(XXx,UUx,'-or')
plt.ylabel('Ux')
plt.xlabel('x')
plt.title('Ux=f(x)')
plt.grid()
plt.show()

#%% CALCUL DE LA DEFO EPSXX
##################################


T1 = np.zeros(n_nodes1)       # Surface 1
EPSXX1 = np.zeros(n_nodes1)   # defo epsxx 1


T2 = np.zeros(n_nodes2)       # Surface 2
EPSXX2 = np.zeros(n_nodes2)   # defo epsxx 2


#boucle sur les elements 

## A VOUS DE JOUER ICI



# Visualisation de la defo
t1 = mtri.Triangulation(xxx1,yyy1,C1)
t2 = mtri.Triangulation(xxx2,yyy2,C2)
plt.figure(3)
plt.triplot(t1)
l  = np.array([  min( np.min(EPSXX1),  np.min(EPSXX2) ),  max( np.max(EPSXX1),  np.max(EPSXX2) )    ])
l =  np.linspace(l[0],l[1], 20)
plt.tricontourf(t1,EPSXX1, levels=l   )
plt.triplot(t2)
plt.tricontourf(t2,EPSXX2,levels=l  )
plt.colorbar()
plt.title('Defo epsxx')
 
 
 
    



 
