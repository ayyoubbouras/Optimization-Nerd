# Importation des libreries necessaires pour la fonctionnement du QuasiNewton

########################
#Les outils mathmatiques
import re
import sympy as sp
import numpy as np
from sympy import Symbol
from scipy.misc import derivative
from numpy import linalg as LA
########################

########################
#le 3d 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numdifftools as nd
########################




zs = [Symbol('x'), Symbol('y')]
global fonction
#################################
#definissions les variables ( 4 var de max ) et le alpha
zs4 = [Symbol('x1'), Symbol('x2'),Symbol('x3'),Symbol('x4')]
zs3 = [Symbol('x1'), Symbol('x2'),Symbol('x3')]
zs2 = [Symbol('x1'), Symbol('x2')]
alpha= Symbol('alpha')
#################################

# On renomme les expressions saisies par l'utilisateur par des expressions deja definies par numpy
replacements = {
    'sin' : 'np.sin',
    'cos' : 'np.cos',
    'exp': 'np.exp',
    'sqrt': 'np.sqrt',
    '^': '**',
    'log':'np.log'
}

# Les mots autorises dans la saisie 
allowed_words = [
    'x',
    'sin',
    'cos',
    'sqrt',
    'exp',
    'log'
]




#La classe du DFP par Le pas exacte
class DFP:
    def __init__(self,f,x0,tol): # Fonction pour intialiser les parametres de la classe
        self.f=f # fonction entree par l'utilisateur
        self.x0=x0 # point initial entree par l'utilisateur
        self.tol=tol # La tolerance entree par l'utilisateur
        self.i=0 # nombre des iterations intialiser par 0
 
    def initial_point(self):  # fonction qui recoit le point intial et le transformer en numpy array
        x0=self.x0
        i=1
        Xs=[]
        k=1
        while (i<len(x0)): # loop pour transformer le x0 d'un string (1,1) en numpy array np.array([1,1])
            if (x0[i]==','):  # si x0[i] == ',' on stocke ce qui est avant la virgule
                nb=float(x0[k:i])
                Xs.append(nb)
                k=i+1
            i=i+1
        Xs.append(float(x0[k:len(x0)-1])) #Exemple : x0 = (1,1,1.5) : Xs.append(float(1)) puis Xs.append(float(1)) puis Xs.append(float(1.5))
        return np.array(Xs)

    def fp_methode(self,f,a,b):# Fonction  de la fausse position pour le cas exceptionnel du newtion if f''(x) = 0
        def f(alpha):  # Evaluer la fonction en 'alpha'
            return eval(f)

        # definir la derive de 1er ordre de la fonction utilise 
        def df(alpha):
            return derivative(f,alpha,0.01) # 0.01 est le pas choisi

        e=abs(a-b)  # l'erreur |a-b|
        xk_1=a
        xk=b
        while(e>=self.tol):  # loop de la methode de la fausse position
            if(df(xk_1)-df(xk)==0): # si df(xk_1) - df(xk) = 0 on retourne xk_1
                return xk_1
            dk=-df(xk)*(xk_1-xk)/(df(xk_1)-df(xk)) # Calcule de la direction chaque iteration
            xk_Plus1=xk+dk
            e=abs(xk_Plus1-xk) # l'erreur pour la condition d'arret
            xk_1=xk
            xk=xk_Plus1

        return xk_Plus1
    #on utilise la methode de newton unidim pour trouver le pas exacte
    def newton_methode(self,fonc ,x0):
        def f(alpha):
            return eval(fonc)

        def df(alpha):
            return derivative(f,alpha,0.01)

        # definir la derive seconde de la fonction utilisee
        def d2f(alpha):
            return derivative(f,alpha,0.1,2)
        xk=x0
        while(True): # loop de la methode de Newton Unidimentionnel
            if(d2f(xk)==0): #Condition qui demande l'appel du Fausse Position
                xk_1=self.fp_methode(fonc,xk,xk+1)
                return xk_1
            xk_1=xk - (df(xk)/d2f(xk))
            k=xk
            xk=xk_1
            if (abs(xk_1-k)<self.tol): # Condition d'Arret
                break

        return xk_1
    # evaluer la fonction entrer comme string selon le nombre des variables entrees
    def fonction1(self,x):
        global n
        n=2
        if n==2: # si le nombre de var = 2
            f=str(eval(self.f, None, dict([z.name, z] for z in zs2)))  # loop pour lire x1 et x2 declarer avant sous formre zs2
            f=f.replace('x1','(x1)') # remplacer x1 par (x1)
            f=f.replace('x1','{x1}') #remplacer x1 par {x1} pour que .format fonctionne
            f=f.replace('x2', '(x2)') # remplacer x2 par (x2)
            f=f.replace('x2', '{x2}') #remplacer x2 par {x2} pour que .format fonctionne
            f=f.format(x1=x[0],x2=x[1]) # utiliser .format pour remplacer la valeur du x[0] dans x1 et x[1] dans x2
        elif(n==3): # si le nombre de var = 3
            f=str(eval(self.f, None, dict([z.name, z] for z in zs3))) # loop pour lire x1 et x2 et x3  declarer avant sous forme zs3
            f=f.replace('x1','(x1)') # remplacer x1 par (x1)
            f=f.replace('x1','{x1}') #remplacer x1 par {x1} pour que .format fonctionne
            f=f.replace('x2', '(x2)') # remplacer x2 par (x2)
            f=f.replace('x2', '{x2}') #remplacer x2 par {x2} pour que .format fonctionne
            f=f.replace('x3','(x3)')  # remplacer x3 par (x3)
            f=f.replace('x3','{x3}') #remplacer x3 par {x3} pour que .format fonctionne
            f=f.format(x1=x[0],x2=x[1],x3=x[2])  # utiliser .format pour remplacer la valeur du x[0] dans x1 et x[1] dans x2 et x[2] dans x3
        else: # si le nombre de var = 4
            f=str(eval(self.f, None, dict([z.name, z] for z in zs4)))  # loop pour lire x1 et x2 et x3 et x4 declarer avant sous formre zs4
            f=f.replace('x1','(x1)') # remplacer x1 par (x1)
            f=f.replace('x1','{x1}') #remplacer x1 par {x1} pour que .format fonctionne
            f=f.replace('x2', '(x2)') # remplacer x2 par (x2)
            f=f.replace('x2', '{x2}') #remplacer x2 par {x2} pour que .format fonctionne
            f=f.replace('x3','(x3)') # remplacer x3 par (x3)
            f=f.replace('x3','{x3}') #remplacer x3 par {x3} pour que .format fonctionne
            f=f.replace('x4','(x4)') # remplacer x4 par (x4)
            f=f.replace('x4','{x4}') #remplacer x4 par {x4} pour que .format fonctionne
            f=f.format(x1=x[0],x2=x[1],x3=x[2],x4=x[3])  # utiliser .format pour remplacer la valeur du x[0] dans x1 et x[1] dans x2 et x[2] dans x3 et x[3] dans x3
        return float(eval(f)) # Retourner evaluation de f en x
    #evalution de la fonction pour le cas de 3d
    def fonction3d(self,x1,x2):
        f=self.string2func() # appel de la fonction string2func()
        return eval(f)

    #fonction pour reformuler la fonction entree en fonction de alpha
    def alphaa(self,x_alpha,alpha):
        n=2 #le cas de n=2 est par defaut
        f=self.f
        f=f.replace('x1','(x1)')  # remplacer x1 par (x1)
        f=f.replace('x1','{x1}') #remplacer x1 par {x1} pour que .format fonctionne
        f=f.replace('x2', '(x2)') # remplacer x2 par (x2)
        f=f.replace('x2', '{x2}') #remplacer x2 par {x2} pour que .format fonctionne
        f=f.format(x1=x_alpha[0],x2=x_alpha[1])  # Utiliser .format pour remplacer la valeur du x_alpha[0] dans x1 et x_alpha[1] dans x2
        if(n==3): #le cas de 3 variables (x1,x2,x3)
            f=f.replace('x3','(x3)')  # remplacer x3 par (x3)
            f=f.replace('x3','{x3}') #remplacer x3 par {x3} pour que .format fonctionne
            f=f.format(x3=x_alpha[2]) # utiliser .format pour remplacer la valeur du x_alpha[2] dans x3 s
        if(n==4):  #le cas de 4 variables (x1,x2,x3,x4)
            f=f.replace('x3','(x3)') #remplacer x3 par (x3)
            f=f.replace('x3','{x3}') #remplacer x3 par {x3} pour que .format fonctionnes
            f=f.replace('x4','(x4)') #remplacer x4 par (x4)
            f=f.replace('x4','{x4}')  #remplacer x4 par {x4} pour que .format fonctionne
            f=f.format(x3=x_alpha[2],x4=x_alpha[3]) # utiliser .format pour remplacer la valeur du x_alpha[2] dans x3 et x_alpha[3] dans x4
        return f
    def string2func(self): # Transformer la fonction d'un string vers une fonction qui peut etre evaluer
        f=self.f
        ''' evaluates the string and returns a function of x '''
        # Trouver tous les mots dans la chaine de caracteres et verifier s'ils sont autorises
        for word in re.findall('[a-zA-Z_]+', f):
            if word not in allowed_words:
                raise ValueError(
                    '"{}" is forbidden to use in math expression'.format(word)
                )

        # On remplace les mots trouves dans la chaine de caracteres
        for old, new in replacements.items():
            f = f.replace(old, new)

        # Eval(fonc) permet d'executer la chaines de caracteres fonc en tant que instruction Python
        return f
    #la fonction de DFP qui appelle le pas de armijo
    def QuasiNewton_DFP(self):
        dict=[]  # intialiser dictionnaire pour stocker les iterationss
        x=self.initial_point() # point initial entree par l'utilisateur
        g=(nd.Gradient(self.fonction1)(x)).T # Calcul du Gradient et l'en mettre en Transposition
        H_k=np.array([[1,0],[0,1]]) # La matrice H Initial
        alpha_k=0.2 # le pas alpha Intial
        list=(self.i,str(x),self.fonction1(x),alpha_k)  #initialiser la liste qui contient (nb d'iteration , x en iteration k , f(x) , le pas )
        dict.insert(self.i,list) # Ajout de la liste en dictionnaire dict[]
        while(True): # loop jusqu'a la realisation du condition d'arret
            self.i+=1  # Incrementation du self.i
            i=self.i # le i initial
            d_k=-H_k.dot(g) #La direction Descente Initiale 
            x_alpha=x-alpha*d_k # x en fonction du symbole 'alpha' exemple x=(1,1) et grad(x) = (2,2) -> x_alpha = (1 - alpha*2 , 1- alpha*2)
            func=self.alphaa(x_alpha,alpha) # Exemple : fonction(x1,x2) = x1 - x2 -> fonction(alpha) = 3-alpha
            alpha_k=self.newton_methode(func ,alpha_k) # Obtention du pas Exacte avec Newton unidimentionnel
            alpha_k=abs(alpha_k)
            xk_PLUS_1=x-abs(alpha_k)*nd.Gradient(self.fonction1)(x)  # Iteration prochaine Avec le Pas Exacte
            xk_PLUS_1=np.array([float(xk_PLUS_1[0]),float(xk_PLUS_1[1])])  # Iteration prochaine Avec le Pas Exacte en float
            list=(i,str(xk_PLUS_1),self.fonction1(xk_PLUS_1),alpha_k) # stocker les iterations sous forme liste
            dict.insert(i,list)  # l'ajout du liste au dictionnaire
            g=(nd.Gradient(self.fonction1)(xk_PLUS_1)).T #Calcul du Gradient en Xk+1
            y_k=nd.Gradient(self.fonction1)(xk_PLUS_1)-nd.Gradient(self.fonction1)(x) # Calcul du Vecteur Y_k
            A_k=((alpha_k)**2*d_k.dot(d_k.T))/((alpha_k)*(d_k.T).dot(y_k))# Calcul de la Matrice A_k
            b=(H_k.dot(y_k)).T # Calcul du vecteur b
            a=-H_k.dot(y_k) # Calcul du vecteur a
            y=((y_k.T).dot(H_k)).dot(y_k)
            B_k=a.dot(b)/y # Calcul de la Matrice B_k
            H_k=H_k+A_k+B_k #Calcul de la matrice H_k+1
            if(LA.norm(xk_PLUS_1-x)<=self.tol): # Condition d'arret
                break
            x=xk_PLUS_1    
        return dict 
    #la fonction de DFP avec inclusion du 3d 
    def QuasiNewton_DFP3d(self):
        x=self.initial_point() # point initial entree par l'utilisateur
        fig=plt.figure() # fig = une figure ou on va afficher le graphe en 3d
        fig.set_size_inches(9, 7, forward=True) # hieght = 9cm et width = 7
        ax=Axes3D(fig, azim=-29, elev=49,auto_add_to_figure=False)
        fig.add_axes(ax) #Ajour du graphe a l'interieur du figure
        X = np.linspace(-3, 3, 30) # Axe des X [-3,3]
        Y = np.linspace(-3, 3, 30) # Axe des Y [-3,3]
        Z = self.fonction3d(X[:,None],Y[None,:]) # Axe des Z= f(X,Y)
        X, Y=np.meshgrid(X, Y)
        ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
        plt.xlabel("Paramètre 1 (x)") # titre de l'axe x
        plt.ylabel("Paramètre 2 (y)") # titre de l'axe y
        g=(nd.Gradient(self.fonction1)(x)).T # d = le Gradient au point intial x0 Transpose
        H_k=np.array([[1,0],[0,1]]) # La matrice H Initial
        alpha_k=1 # Le pas Initial
        ax.scatter(x[0], x[1], self.fonction1(x), marker='o', s=10, color='#00FF00')
        plt.draw()
        while(True): # loop jusqu'a la realisation du condition d'arret
            self.i+=1 # Incrementation du self.i
            i=self.i # le i initial
            d_k=-H_k.dot(g) #La direction Descente Initiale 
            x_alpha=x-alpha*d_k # x en fonction du symbole 'alpha' exemple x=(1,1) et grad(x) = (2,2) -> x_alpha = (1 - alpha*2 , 1- alpha*2)
            func=self.alphaa(x_alpha,alpha) # Exemple : fonction(x1,x2) = x1 - x2 -> fonction(alpha) = 3-alpha
            alpha_k=self.newton_methode(func ,alpha_k) # Obtention du pas Exacte avec Newton unidimentionnel
            alpha_k=abs(alpha_k)
            xk_PLUS_1=x-abs(alpha_k)*nd.Gradient(self.fonction1)(x) # Iteration prochaine Avec le Pas Exacte
            xk_PLUS_1=np.array([float(xk_PLUS_1[0]),float(xk_PLUS_1[1])]) # Iteration prochaine Avec le Pas Exacte en float
            g=(nd.Gradient(self.fonction1)(xk_PLUS_1)).T #Calcul du Gradient en Xk+1
            y_k=nd.Gradient(self.fonction1)(xk_PLUS_1)-nd.Gradient(self.fonction1)(x) # Calcul du Vecteur Y_k
            A_k=((alpha_k)**2*d_k.dot(d_k.T))/((alpha_k)*(d_k.T).dot(y_k)) # Calcul de la Matrice A_k
            b=(H_k.dot(y_k)).T # Calcul du vecteur b
            a=-H_k.dot(y_k) # Calcul du vecteur a
            y=((y_k.T).dot(H_k)).dot(y_k)
            B_k=a.dot(b)/y # Calcul de la Matrice B_k
            H_k=H_k+A_k+B_k #Calcul de la matrice H_k+1
            ########################################################
            #Le dessin des points des iterations chaque 0.05 seconde
            ax.scatter(xk_PLUS_1[0], xk_PLUS_1[1], self.fonction1(xk_PLUS_1), marker='o', s=10, color='#00FF00')
            plt.draw()
            plt.pause(0.05)
            ########################################################
            if(LA.norm(xk_PLUS_1-x)<=self.tol): # Condition d'arret
                break
            x=xk_PLUS_1    
        return dict 

#La classe du BFGS par Le pas exacte
class BFGS:
    def __init__(self,f,x0,tol): # Fonction pour intialiser les parametres de la classe
        self.f=f # fonction entree par l'utilisateur
        self.x0=x0 # point initial entree par l'utilisateur
        self.tol=tol # La tolerance entree par l'utilisateur
        self.i=0 # nombre des iterations intialiser par 0
    def initial_point(self): # fonction qui recoit le point intial et le transformer en numpy array
        x0=self.x0
        i=1
        Xs=[]
        k=1
        while (i<len(x0)): # loop pour transformer le x0 d'un string (1,1) en numpy array np.array([1,1])
            if (x0[i]==','): # si x0[i] == ',' on stocke ce qui est avant la virgule
                nb=float(x0[k:i])
                Xs.append(nb)
                k=i+1
            i=i+1
        Xs.append(float(x0[k:len(x0)-1])) # Exemple : x0 = (1,1,1.5) : Xs.append(float(1)) puis Xs.append(float(1)) puis Xs.append(float(1.5))
        return np.array(Xs)
    def string2func(self): # Transformer la fonction d'un string vers une fonction qui peut etre evaluer
        f=self.f
        ''' evaluates the string and returns a function of x '''
        # Trouver tous les mots dans la chaine de caracteres et verifier s'ils sont autorises
        for word in re.findall('[a-zA-Z_]+', f):
            if word not in allowed_words:
                raise ValueError(
                    '"{}" is forbidden to use in math expression'.format(word)
                )

        # On remplace les mots trouves dans la chaine de caracteres
        for old, new in replacements.items():
            f = f.replace(old, new)

        # Eval(fonc) permet d'executer la chaines de caracteres fonc en tant que instruction Python
        return f
    
    def fonction1(self,x): # evaluer la fonction entree comme string selon le nombre des variables entrées
        global n
        n=2
        if n==2: # si le nombre de var = 2
            f=str(eval(self.f, None, dict([z.name, z] for z in zs2))) # loop pour lire x1 et x2 declarer avant sous formre zs2
            f=f.replace('x1','(x1)')  # remplacer x1 par (x1)
            f=f.replace('x1','{x1}') #remplacer x1 par {x1} pour que .format fonctionne
            f=f.replace('x2', '(x2)') # remplacer x2 par (x2)
            f=f.replace('x2', '{x2}') #remplacer x2 par {x2} pour que .format fonctionne
            f=f.format(x1=x[0],x2=x[1]) # utiliser .format pour remplacer la valeur du x[0] dans x1 et x[1] dans x2
        elif(n==3):  # si le nombre de var = 3
            f=str(eval(self.f, None, dict([z.name, z] for z in zs3)))  # loop pour lire x1 et x2 et x3  declarer avant sous forme zs3
            f=f.replace('x1','(x1)') # remplacer x1 par (x1)
            f=f.replace('x1','{x1}') #remplacer x1 par {x1} pour que .format fonctionne
            f=f.replace('x2', '(x2)') # remplacer x2 par (x2)
            f=f.replace('x2', '{x2}') #remplacer x2 par {x2} pour que .format fonctionne
            f=f.replace('x3','(x3)') # remplacer x3 par (x3)
            f=f.replace('x3','{x3}')  #remplacer x3 par {x3} pour que .format fonctionne
            f=f.format(x1=x[0],x2=x[1],x3=x[2]) # utiliser .format pour remplacer la valeur du x[0] dans x1 et x[1] dans x2 et x[2] dans x3
        else:
            f=str(eval(self.f, None, dict([z.name, z] for z in zs4))) # loop pour lire x1 et x2 et x3 et x4 declarer avant sous formre zs4
            f=f.replace('x1','(x1)') # remplacer x1 par (x1)
            f=f.replace('x1','{x1}') #remplacer x1 par {x1} pour que .format fonctionne
            f=f.replace('x2', '(x2)')  # remplacer x2 par (x2)
            f=f.replace('x2', '{x2}') #remplacer x2 par {x2} pour que .format fonctionne
            f=f.replace('x3','(x3)') # remplacer x3 par (x3)
            f=f.replace('x3','{x3}')  #remplacer x3 par {x3} pour que .format fonctionne
            f=f.replace('x4','(x4)') # remplacer x4 par (x4)
            f=f.replace('x4','{x4}') #remplacer x4 par {x4} pour que .format fonctionne
            f=f.format(x1=x[0],x2=x[1],x3=x[2],x4=x[3]) # utiliser .format pour remplacer la valeur du x[0] dans x1 et x[1] dans x2 et x[2] dans x3 et x[3] dans x3
        return float(eval(f)) # Retourner evaluation de f en x
    def find_interval(self,x0,dk): # Trouver l'intervalle des alpha avec dk = direction descente
        left = 0 # [left=0 ,right =1]
        step = 1
        right = step + left
        k = 2
        while True: # loop while fonction(x0 + right * dk) < self.fonction1(x0 + left * dk)
            if self.fonction1(x0 + right * dk) < self.fonction1(x0 + left * dk):
                step *= k # step = step * k
                left = right
                right = left + step
                return left, right # sortie de la loop
            else:
                if right <= 0: # right ne doit pas etre <= 0
                    left = right
                    right = 0
                return left, right
    def find_root(self,x0, dk, a, b): # Fonction pour trouver le root ou le pas a choisir a partir d'interval retourner par fin_interval()
        alpha1 = (a + b) / 2          # c'est juste comme la bissection 
        rho = 0.01                    # avec dk est la direction de descente du BFGS
        t = 3
        xk = x0
        gk = nd.Gradient(self.fonction1)(xk) # le Gradient en point xk
        left = a
        right = b
        while True: # loop while  fonction(xk + alpha1 * dk) - self.fonction1(xk) <= rho * alpha1 * np.dot(gk, dk)
            if self.fonction1(xk + alpha1 * dk) - self.fonction1(xk) <= rho * alpha1 * np.dot(gk, dk):#Goldstein
                if self.fonction1(xk + alpha1 * dk) - self.fonction1(xk) >= (1 - rho) * alpha1 * np.dot(gk, dk):
                    return alpha1
                else:
                    left = alpha1
                    right = right
                    if right >= b:
                        alpha1 = t * alpha1
                        return alpha1
                    alpha1 = (left + right) / 2
            else:
                left = left
                right = alpha1
                alpha1 = (left + right) / 2
            return alpha1
    def BFGS_method(self):
        dict=[] # intialiser dictionnaire pour stocker les iterations
        self.i= 0 #  initialiser le i par 0 (nombre des iterations )
        xk = self.initial_point() # point initial entree par l'utilisateur
        shape = np.shape(xk)[0] # shape = le nombre de variable , Exemple : xk =(1,1,1,1) -> shape = 4
        gk = nd.Gradient(self.fonction1)(xk) # Calcul du Gradient en point xk
        Gk = nd.Hessian(self.fonction1)(xk) #Calcul du Hessian en point xk
        dk = -1 * np.dot(np.linalg.inv(Gk), gk) # Calcul de la direction de descente initial du BFGS
        alpha1=1 # Intialisation du pas par 1
        while np.linalg.norm(gk) > self.tol and self.i < 500: # 2 conditions d'arret ||gk|| < tolerance et nombre des iterations >= 500
            sk = alpha1 * dk # Multiplication du pas par la direction du descente
            xk = xk + sk # Iteration prochaine Avec le Pas Exacte
            list=(self.i,str(xk),self.fonction1(xk),alpha1) # stocker les iterations sous forme liste
            dict.insert(self.i,list) # l'Ajout du liste au dictionnaire
            alpha1 = self.find_root(xk, dk, self.find_interval(xk, dk)[0], self.find_interval(xk, dk)[1]) # Appel de la fonction find_root pour trouver le pas exacte
            self.i += 1 # Incrementation du self.i
            sk = sk.reshape(shape, 1) # reformer le sk pour etre sous forme vecteur vertical
            yk = nd.Gradient(self.fonction1)(xk) - gk # Calcul du vecteur yk qui est le Gradient en xk+1 moins le gradient en xk
            yk = yk.reshape(shape, 1) # reformer le yk pour etre sous forme vecteur vertical
            Gk = Gk + np.dot(yk, yk.T)/np.dot(yk.T, sk) - Gk.dot(sk).dot(sk.T).dot(Gk)/sk.T.dot(Gk).dot(sk) # Calcul du Gk
            gk = nd.Gradient(self.fonction1)(xk) # Calcul du vecteur yk qui est le Gradient en xk 
            dk = -1 * np.dot(np.linalg.inv(Gk), gk) # Calcul du dk ( la direction de descente de BFGS )
        return dict
     #evalution de la fonction pour le cas de 3d
    def fonction3d(self,x1,x2):
        f=self.string2func()
        return eval(f)
    #BFGS par le pas exacte
    def BFGS3d(self):
        xk=self.initial_point() # point initial entree par l'utilisateur
        fig=plt.figure() # fig = une figure ou on va afficher le graphe en 3d
        fig.set_size_inches(9, 7, forward=True) # hieght = 9cm et width = 7
        ax=Axes3D(fig, azim=-29, elev=49) #Ajout du graphe a l'interieur du figure 
        X = np.linspace(-3, 3, 30) # Axe des X [-3,3] avec 30 points entre cet interval
        Y = np.linspace(-3, 3, 30) # Axe des Y [-3,3] avec 30 points entre cet interval
        Z = self.fonction3d(X[:,None],Y[None,:]) # Axe des Z= f(X,Y)
        X, Y=np.meshgrid(X, Y) # Axe des Z= f(X,Y)
        ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
        plt.xlabel("Paramètre 1 (x)") # le titre d'axe des x
        plt.ylabel("Paramètre 2 (y)") # le titre d'axe des y
        shape = np.shape(xk)[0]  # shape = le nombre de variable , Exemple : xk =(1,1,1,1) -> shape = 4
        gk = nd.Gradient(self.fonction1)(xk) # Calcul du Grdient en point xk+1
        Gk = nd.Hessian(self.fonction1)(xk)  #Calcul du Hessian en point xk
        dk = -1 * np.dot(np.linalg.inv(Gk), gk) # Calcul de la direction de descente initial du BFGS
        alpha_k=1 # Intialisation du pas par 1
        while(True):
            alpha1 = self.find_root(xk, dk, self.find_interval(xk, dk)[0], self.find_interval(xk, dk)[1]) # Appel de la fonction find_root pour trouver le pas exacte
            sk = alpha1 * dk # Multiplication du pas par la direction du descente
            xk = xk + sk # Iteration prochaine Avec le Pas Exacte
            sk = sk.reshape(shape, 1) # reformer le sk pour etre sous forme vecteur vertical
            yk = nd.Gradient(self.fonction1)(xk) - gk # Calcul du vecteur yk qui est le Gradient en xk+1 moins le gradient en xk
            yk = yk.reshape(shape, 1)  # reformer le yk pour etre sous forme vecteur vertical
            Gk = Gk + np.dot(yk, yk.T)/np.dot(yk.T, sk) - Gk.dot(sk).dot(sk.T).dot(Gk)/sk.T.dot(Gk).dot(sk) # Calcul du Gk
            gk = nd.Gradient(self.fonction1)(xk)  # Calcul du Grdient en point xk+1
            dk = -1 * np.dot(np.linalg.inv(Gk), gk)  # Calcul du dk ( la direction de descente de BFGS )
            ################################# Les iterations chaque 0.05 secondes en 3d #########################
            ax.scatter(xk[0], xk[1], self.fonction1(xk), marker='o', s=10, color='#00FF00')
            plt.draw()
            plt.pause(0.05)
            #####################################################################################################
            if(np.linalg.norm(gk)<self.tol): #Condition d'arret ||gk|| < tolerance 
                break

#La classe du SR1 par Le pas exacte
class SR1:
    def __init__(self,f,x0,tol): # Fonction pour intialiser les parametres de la classe
        self.f=f # fonction entree par l'utilisateur
        self.x0=x0 # point initial entree par l'utilisateur
        self.tol=tol # La tolerance entree par l'utilisateur
        self.i=0 # nombre des iterations intialiser par 0
    #la fonction pour transformer le point initial d'un string au numpy array
    def initial_point(self): 
        x0=self.x0
        i=1
        Xs=[]
        k=1
        while (i<len(x0)): # loop pour transformer le x0 d'un string (1,1) en numpy array np.array([1,1])
            if (x0[i]==','):  # si x0[i] == ',' on stocke ce qui est avant la virgule
                nb=float(x0[k:i])
                Xs.append(nb)
                k=i+1
            i=i+1
        Xs.append(float(x0[k:len(x0)-1])) # Exemple : x0 = (1,1,1.5) : Xs.append(float(1)) puis Xs.append(float(1)) puis Xs.append(float(1.5))
        return np.array(Xs)
    #la fonction pour transformer la fonction entree d'un string en une fonction qui peut etre evalue
    def string2func(self):
        f=self.f
        ''' evaluates the string and returns a function of x '''
        # Trouver tous les mots dans la chaine de caracteres et verifier s'ils sont autorises
        for word in re.findall('[a-zA-Z_]+', f):
            if word not in allowed_words:
                raise ValueError(
                    '"{}" is forbidden to use in math expression'.format(word)
                )

        # On remplace les mots trouves dans la chaine de caracteres
        for old, new in replacements.items():
            f = f.replace(old, new)

        # Eval(fonc) permet d'executer la chaines de caracteres fonc en tant que instruction Python
        return f
    # evaluer la fonction entrer comme string selon le nombre des variables entrees
    def fonction1(self,x):
        global n
        n=2
        if n==2: # si le nombre de var = 2
            f=str(eval(self.f, None, dict([z.name, z] for z in zs2))) # loop pour lire x1 et x2 declarer avant sous formre zs2
            f=f.replace('x1','(x1)') # remplacer x1 par (x1)
            f=f.replace('x1','{x1}')  #remplacer x1 par {x1} pour que .format fonctionne
            f=f.replace('x2', '(x2)') # remplacer x2 par (x2)
            f=f.replace('x2', '{x2}') #remplacer x2 par {x2} pour que .format fonctionne
            f=f.format(x1=x[0],x2=x[1]) # utiliser .format pour remplacer la valeur du x[0] dans x1 et x[1] dans x2
        elif(n==3): # si le nombre de var = 3
            f=str(eval(self.f, None, dict([z.name, z] for z in zs3))) # loop pour lire x1 et x2 et x3  declarer avant sous forme zs3
            f=f.replace('x1','(x1)') # remplacer x1 par (x1)
            f=f.replace('x1','{x1}') #remplacer x1 par {x1} pour que .format fonctionne
            f=f.replace('x2', '(x2)') # remplacer x2 par (x2)
            f=f.replace('x2', '{x2}') #remplacer x2 par {x2} pour que .format fonctionne
            f=f.replace('x3','(x3)') # remplacer x3 par (x3)
            f=f.replace('x3','{x3}') #remplacer x3 par {x3} pour que .format fonctionne
            f=f.format(x1=x[0],x2=x[1],x3=x[2]) # utiliser .format pour remplacer la valeur du x[0] dans x1 et x[1] dans x2 et x[2] dans x3 
        else: #cas n=4
            f=str(eval(self.f, None, dict([z.name, z] for z in zs4))) # loop pour lire x1 et x2 et x3 et x4 declarer avant sous formre zs4
            f=f.replace('x1','(x1)') # remplacer x1 par (x1)
            f=f.replace('x1','{x1}') #remplacer x1 par {x1} pour que .format fonctionne
            f=f.replace('x2', '(x2)') # remplacer x2 par (x2)
            f=f.replace('x2', '{x2}') #remplacer x2 par {x2} pour que .format fonctionne
            f=f.replace('x3','(x3)') # remplacer x3 par (x3)
            f=f.replace('x3','{x3}') #remplacer x3 par {x3} pour que .format fonctionne
            f=f.replace('x4','(x4)') # remplacer x4 par (x4)
            f=f.replace('x4','{x4}') #remplacer x4 par {x4} pour que .format fonctionne
            f=f.format(x1=x[0],x2=x[1],x3=x[2],x4=x[3]) # utiliser .format pour remplacer la valeur du x[0] dans x1 et x[1] dans x2 et x[2] dans x3 et x[3] dans x3
        return float(eval(f)) # Retourner evaluation de f en x
    def find_interval(self,x0,dk):              
        left = 0
        step = 1
        right = step + left
        k = 2
        while True:
            if self.fonction1(x0 + right * dk) < self.fonction1(x0 + left * dk): # loop while  fonction(xk + alpha1 * dk) - self.fonction1(xk) <= rho * alpha1 * np.dot(gk, dk)
                step *= k # step = step * k
                left = right
                right = left + step
                return left, right # sortie de la loop
            else:
                if right <= 0:
                    left = right
                    right = 0
                return left, right
    def find_root(self,x0, dk, a, b):        # Fonction pour trouver le root ou le pas a choisir a partir d'interval retourner par fin_interval()
        alpha1 = (a + b) / 2                 # c'est juste comme la bissection 
        rho = 0.01                           # avec dk est la direction de descente du BFGS
        t = 3
        xk = x0
        gk = nd.Gradient(self.fonction1)(xk)
        left = a
        right = b
        while True: # loop while  fonction(xk + alpha1 * dk) - self.fonction1(xk) <= rho * alpha1 * np.dot(gk, dk)
            if self.fonction1(xk + alpha1 * dk) - self.fonction1(xk) <= rho * alpha1 * np.dot(gk, dk):#Goldstein
                if self.fonction1(xk + alpha1 * dk) - self.fonction1(xk) >= (1 - rho) * alpha1 * np.dot(gk, dk):
                    return alpha1
                else:
                    left = alpha1
                    right = right
                    if right >= b:
                        alpha1 = t * alpha1
                        return alpha1
                    alpha1 = (left + right) / 2
            else:
                left = left
                right = alpha1
                alpha1 = (left + right) / 2
            return alpha1
    def SR1(self):
        dict=[] # intialiser dictionnaire pour stocker les iterations
        self.i = 0
        xk = self.initial_point() # point initial entree par l'utilisateur
        shape = np.shape(xk)[0] # shape = le nombre de variable , Exemple : xk =(1,1,1,1) -> shape = 4
        gk = nd.Gradient(self.fonction1)(xk) # Calcul du Gradient en point xk
        Gk = nd.Hessian(self.fonction1)(xk) # Calcul du Hessian en point xk
        dk = -1 * np.dot(Gk, gk) # Calcul de la direction descente 
        while np.linalg.norm(gk) > self.tol and self.i < 500: # 2 conditions d'arret ||gk|| < tolerance et nombre des iterations >= 500
            alpha1 = self.find_root(xk, dk, self.find_interval(xk, dk)[0], self.find_interval(xk, dk)[1]) # Appel de la fonction find_root pour trouver le pas exacte
            #w[:, step] = np.transpose(xk)
            self.i += 1
            sk = alpha1 * dk # Multiplication du pas par la direction du descente
            xk = xk + sk # Iteration prochaine Avec le Pas Exacte
            xkPrecision="({0:.4f},{1:.4f})".format(xk[0],xk[1]) # Pour avoir juste 4 chiffres apres la virgule
            list=(self.i,str(xkPrecision),self.fonction1(xk),alpha1)  # stocker les iterations sous forme liste
            dict.insert(self.i,list) # l'Ajout du liste au dictionnaire
            sk = sk.reshape(shape, 1) # reformer le sk pour etre sous forme vecteur vertical
            gk2=nd.Gradient(self.fonction1)(xk) # Calcul du vecteur yk qui est le Gradient en xk+1 
            yk = gk2 - gk # Calcul du vecteur yk qui est le Gradient en xk+1 moins grandient en xk
            yk = yk.reshape(shape, 1) # reformer le yk pour etre sous forme vecteur vertical
            Gk = Gk + ((sk - np.dot(Gk, yk)).dot((sk - np.dot(Gk, yk)).T) / ((sk - np.dot(Gk, yk)).T).dot(yk)) # Calcul du Gk
            gk = gk2
            dk = -1 * np.dot(Gk, gk) # Calcul de la direction de descente
        return dict  
     #evalution de la fonction pour le cas de 3d
    def fonction3d(self,x1,x2):
        f=self.string2func()
        return eval(f)
    #SR1 en 3d
    def SR1_3d(self):
        pass
        '''
        global fonction,func
        xk=self.initial_point()
        fig=plt.figure()
        fig.set_size_inches(9, 7, forward=True)
        ax=Axes3D(fig, azim=-29, elev=49)
        X = np.linspace(-3, 3, 30)
        Y = np.linspace(-3, 3, 30)
        Z = self.fonction3d(X[:,None],Y[None,:])
        X, Y=np.meshgrid(X, Y)
        ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
        plt.xlabel("Paramètre 1 (x)")
        plt.ylabel("Paramètre 2 (y)")
        self.i = 0
        shape = np.shape(xk)[0]
        gk = nd.Gradient(self.fonction1)(xk)
        Gk = nd.Hessian(self.fonction1)(xk)
        dk = -1 * np.dot(Gk, gk)
        while(True):
            alpha1 = self.find_root(xk, dk, self.find_interval(xk, dk)[0], self.find_interval(xk, dk)[1])
            #w[:, step] = np.transpose(xk)
            self.i += 1
            sk = alpha1 * dk
            xk = xk + sk
            sk = sk.reshape(shape, 1)
            gk2=nd.Gradient(self.fonction1)(xk)
            yk = gk2 - gk
            yk = yk.reshape(shape, 1)
            Gk = Gk + ((sk - np.dot(Gk, yk)).dot((sk - np.dot(Gk, yk)).T) / ((sk - np.dot(Gk, yk)).T).dot(yk))
            gk = gk2
            dk = -1 * np.dot(Gk, gk)
            ax.scatter(xk[0], xk[1], self.fonction1(xk), marker='o', s=10, color='#00FF00')
            plt.draw()
            plt.pause(0.05)
            if(np.linalg.norm(gk)<self.tol):
                break
       '''





