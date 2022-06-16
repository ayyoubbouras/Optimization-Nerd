
# Importation des libreries necessaires pour le fonctionnement de Gradient descent
########################
#le 3d 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
########################

########################
#Les outils mathematiques
import numpy as np
from sympy import *
from scipy.misc import derivative
from numpy import linalg as LA
import numdifftools as nd
from sympy import Symbol
import re
#########################

#####################################################
#definissons les variables comme de symboles a reconnaitre ( 4 var de max ) et le alpha
zs4 = [Symbol('x1'), Symbol('x2'),Symbol('x3'),Symbol('x4')]
zs3 = [Symbol('x1'), Symbol('x2'),Symbol('x3')]
zs2 = [Symbol('x1'), Symbol('x2')]
alpha= Symbol('alpha')
#####################################################


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

#La classe du gradient par le pas exacte
class gradient_descent:
    def __init__(self,f,x0,tol,nv): # Fonction pour intialiser les parametres de la classe
        self.nv=nv # nombre de variable entree par l'utilisateur
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
    def fp_methode(self,fa,a,b): # Fonction  de la fausse position pour le cas exceptionnel du newtion if f''(x) = 0
        def f(alpha): # Evaluer la fonction en 'alpha'
            return eval(fa)

        # definir la derive de 1er ordre de la fonction utilise 
        def df(alpha):
            return derivative(f,alpha,0.01) # 0.01 est le pas choisi

        e=abs(a-b) # l'erreur |a-b|
        xk_1=a
        xk=b
        while(e>=self.tol): # loop de la methode de la fausse position
            if(df(xk_1)-df(xk)==0): # si df(xk_1) - df(xk) = 0 on retourne xk_1
                return xk_1
            dk=-df(xk)*(xk_1-xk)/(df(xk_1)-df(xk)) # Calcule de la direction chaque iteration
            xk_Plus1=xk+dk
            e=abs(xk_Plus1-xk)  # l'erreur pour la condition d'arret
            xk_1=xk
            xk=xk_Plus1

        return xk_Plus1
    # Transformer la fonction d'un string vers une fonction qui peut etre evaluer
    def string2func(self):
        f=self.f
        ''' evaluates the string and returns a function of x '''
        # Trouver tous les mots dans la chaine de caracteres et verifier s'ils sont autorises
        for word in re.findall('[a-zA-Z_]+', f):
            if word not in allowed_words: # allowed_words est declare au debut du ce module
                raise ValueError(
                    '"{}" is forbidden to use in math expression'.format(word)
                )

        # On remplace les mots trouves dans la chaine de caracteres
        for old, new in replacements.items():
            f = f.replace(old, new)

        # Eval(fonc) permet d'executer la chaines de caracteres fonc en tant que instruction Python
        return f
    #on utilise la methode de newton unidim pour trouver le pas exacte
    def newton_methode(self,fonc ,x0):
        t=self.tol #la tolerance
        def f(alpha): # lire la fonction qui est en fonction de 'alpha'
            return eval(fonc)

        def df(alpha): # Trouver la derivee d'un h=0.01
            return derivative(f,alpha,0.01)

        # definir la derive seconde de la fonction utilisee
        def d2f(alpha):
            return derivative(f,alpha,0.01,2)
        xk=x0
        while(True): #loop de la fonction de newton unidimentionnel
            if(d2f(xk)==0):
                xk_1=self.fp_methode(fonc,xk,xk+1)
                return xk_1
            xk_1=xk - (df(xk)/d2f(xk))
            k=xk
            xk=xk_1
            if (abs(xk_1-k)<t): # condition d'arret
                break

        return xk_1

    # evaluer la fonction entree comme string selon le nombre des variables entrées
    def fonction1(self,x):
        if self.nv==2: # si le nombre de var = 2
            f=str(eval(self.f, None, dict([z.name, z] for z in zs2))) # loop pour lire x1 et x2 declarer avant sous formre zs2
            f=f.replace('x1','(x1)') # remplacer x1 par (x1)
            f=f.replace('x1','{x1}') #remplacer x1 par {x1} pour que .format fonctionne
            f=f.replace('x2', '(x2)') # remplacer x2 par (x2)
            f=f.replace('x2', '{x2}') #remplacer x2 par {x2} pour que .format fonctionne
            f=f.format(x1=x[0],x2=x[1]) # utiliser .format pour remplacer la valeur du x[0] dans x1 et x[1] dans x2
        elif(self.nv==3):  # si le nombre de var = 3
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
            f=f.replace('x2', '(x2)')  # remplacer x2 par (x2)
            f=f.replace('x2', '{x2}') #remplacer x2 par {x2} pour que .format fonctionne
            f=f.replace('x3','(x3)') # remplacer x3 par (x3)
            f=f.replace('x3','{x3}') #remplacer x3 par {x3} pour que .format fonctionne
            f=f.replace('x4','(x4)')  # remplacer x4 par (x4)
            f=f.replace('x4','{x4}')  #remplacer x4 par {x4} pour que .format fonctionne
            f=f.format(x1=x[0],x2=x[1],x3=x[2],x4=x[3]) # utiliser .format pour remplacer la valeur du x[0] dans x1 et x[1] dans x2 et x[2] dans x3 et x[3] dans x3
        return np.float64(float(eval(f))) # Retourner evaluation de f en x
    #la fonction d'evalution pour le 3d
    def fonction3d(self,x1,x2):
        f=self.string2func()
        return eval(f)

    #fonction pour reformuler la fonction entree en fonction de alpha 
    def alphaa(self,x_alpha,alpha):
        #le cas de n=2 est par defaut
        f=self.f
        f=f.replace('x1','(x1)') # remplacer x1 par (x1)
        f=f.replace('x1','{x1}') #remplacer x1 par {x1} pour que .format fonctionne
        f=f.replace('x2', '(x2)') # remplacer x2 par (x2)
        f=f.replace('x2', '{x2}') #remplacer x2 par {x2} pour que .format fonctionne
        f=f.format(x1=x_alpha[0],x2=x_alpha[1]) # utiliser .format pour remplacer la valeur du x_alpha[0] dans x1 et x_alpha[1] dans x2
        if(self.nv==3): #le cas de 3 variables (x1,x2,x3)
            f=f.replace('x3','(x3)')  # remplacer x3 par (x3)
            f=f.replace('x3','{x3}')  #remplacer x3 par {x3} pour que .format fonctionne
            f=f.format(x3=x_alpha[2]) # utiliser .format pour remplacer la valeur du x_alpha[2] dans x3 
        if(self.nv==4): #le cas de 4 variables (x1,x2,x3,x4)
            f=f.replace('x3','(x3)') #remplacer x3 par (x3)
            f=f.replace('x3','{x3}') #remplacer x3 par {x3} pour que .format fonctionne
            f=f.replace('x4','(x4)') #remplacer x4 par (x4)
            f=f.replace('x4','{x4}') #remplacer x4 par {x4} pour que .format fonctionne
            f=f.format(x3=x_alpha[2],x4=x_alpha[3]) # utiliser .format pour remplacer la valeur du x_alpha[2] dans x3 et x_alpha[3] dans x4
        return f
    #la fonction de gradient qui appelle le pas exacte
    def GradientDescent(self):
        dict=[] # intialiser dictionnaire pour stocker les iterations
        x=self.initial_point() # point initial entree par l'utilisateur
        alpha_k=0.2 # Initialiser le pas alpha
        list=(self.i,str(x),self.fonction1(x),alpha_k) #initialiser la liste qui contient (nb d'iteration , x en iteration k , f(x) , le pas )
        dict.insert(self.i,list) # Ajout de la liste en dictionnaire dict[]
        while(True): # loop jusqu'a la realisation du condition d'arret
            self.i+=1 # Incrementation du self.i
            i=self.i
            x_alpha=x-alpha*nd.Gradient(self.fonction1)(x) # x en fonction du symbole 'alpha' exemple x=(1,1) et grad(x) = (2,2) -> x_alpha = (1 - alpha*2 , 1- alpha*2)
            fc=self.alphaa(x_alpha,alpha) # la fonction en fonction de alpha
            alpha_k=self.newton_methode(fc ,alpha_k) # troucher le pas exacte par newton
            alpha_k=abs(alpha_k)
            xk_PLUS_1=x-alpha_k*nd.Gradient(self.fonction1)(x) # Iteration prochaine Avec le Pas Exacte
            ############## Etudier les cas de nombre de variables entree par l'utilisateur ####################
            if self.nv==2: # Cas : nombre de variable = 2
                xk_PLUS_1=np.array([float(xk_PLUS_1[0]),float(xk_PLUS_1[1])])
                xk_str="({0:.4f},{1:.4f})".format(xk_PLUS_1[0],xk_PLUS_1[1])
            elif self.nv==3: # Cas : nombre de variable = 3
                xk_PLUS_1=np.array([float(xk_PLUS_1[0]),float(xk_PLUS_1[1]),float(xk_PLUS_1[2])])
                xk_str="({0:.4f},{1:.4f},{2:.4f})".format(xk_PLUS_1[0],xk_PLUS_1[1],xk_PLUS_1[2])
            elif self.nv==4: # Cas : nombre de variable = 4
                xk_PLUS_1=np.array([float(xk_PLUS_1[0]),float(xk_PLUS_1[1]),float(xk_PLUS_1[2]),float(xk_PLUS_1[3])])
                xk_str="({0:.4f},{1:.4f},{2:.4f})".format(xk_PLUS_1[0],xk_PLUS_1[1],xk_PLUS_1[2],xk_PLUS_1[3])
            list=(i,xk_str,self.fonction1(xk_PLUS_1),alpha_k)  # stocker les iterations sous forme liste
            dict.insert(i,list) # l'ajout du liste au dictionnaire
            if(LA.norm(xk_PLUS_1-x)<self.tol): #condition d'arret
                break
            x=xk_PLUS_1
        return dict
    # la fonction pour obtenir la fleche du direction de descente
    def get_arrow(self,c,d):
        x = c[0] # les coordinations de la position (x,y,z)
        y = c[1] # les coordinations de la position (x,y,z)
        z = self.fonction1(c) # les coordinations de la position (x,y,z)
        u,v= d[0],d[1] # Gradient direction par rapport au plan (x,y)
        w = self.fonction1(c) # l'image de la direction 
        return x,y,z,u/40,v/40,w/3   

    #la fonction de gradient par le pas exacte en 3d
    def GradientDescent3d(self):
        x=self.initial_point() # point initial entree par l'utilisateur
        d=-nd.Gradient(self.fonction1)(x) # d = le Gradient au point intial x0
        
        fig=plt.figure() # fig = une figure ou on va afficher le graphe en 3d
        fig.set_size_inches(9, 7, forward=True) # hieght = 9cm et width = 7
        ax=Axes3D(fig, azim=-29, elev=49,auto_add_to_figure=False)
        fig.add_axes(ax)  #Ajout du graphe a l'interieur du figure 
        X = np.linspace(-3, 3, 30) # Axe des X [-3,3] avec 30 points entre cet interval
        Y = np.linspace(-3, 3, 30) # Axe des Y [-3,3] avec 30 points entre cet interval
        Z = self.fonction3d(X[:,None],Y[None,:]) # Axe des Z= f(X,Y)
        X, Y=np.meshgrid(X, Y) 
        ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
        ax.quiver(*(self.get_arrow(x,d)))

        plt.xlabel("Paramètre 1 (x)") # titre des axes x
        plt.ylabel("Paramètre 2 (y)") # titre des axes y
 
        alpha_k=0.2
        while(True): # loop jusqu'a la realisation du condition d'arret
            x_alpha=x-alpha*nd.Gradient(self.fonction1)(x) # x en fonction du symbole 'alpha' exemple x=(1,1) et grad(x) = (2,2) -> x_alpha = (1 - alpha*2 , 1- alpha*2)
            func=self.alphaa(x_alpha,alpha) # la fonction en fonction de alpha
            alpha_k=self.newton_methode(func ,alpha_k) # troucher le pas exacte par newton
            alpha_k=abs(alpha_k)
            xk_PLUS_1=x-alpha_k*nd.Gradient(self.fonction1)(x)  # Iteration prochaine Avec le Pas Exacte
            xk_PLUS_1=np.array([float(xk_PLUS_1[0]),float(xk_PLUS_1[1])])  # Iteration prochaine Avec le Pas Exacte  en float
            x=xk_PLUS_1
            d=-nd.Gradient(self.fonction1)(x) # Calcul du Gradient au point x
            ax.scatter(xk_PLUS_1[0], xk_PLUS_1[1], self.fonction1(xk_PLUS_1), marker='o', s=10, color='#00FF00') 
            ax.quiver(*(self.get_arrow(x,d)))
            plt.draw() # Dessiner les iteration en 3d
            plt.pause(0.05) # Ieration chaque 0.05 second
            #if(LA.norm(xk_PLUS_1-x)<self.tol): # condition d'arret
                #break
           





