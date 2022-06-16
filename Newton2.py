

import numpy as np
import re
import random
from numpy.linalg import eig
from numpy import linalg as la
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from sympy import *
from scipy.misc import derivative
import numdifftools as nd
from sympy import Symbol
import re



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

zs4 = [Symbol('x1'), Symbol('x2'),Symbol('x3'),Symbol('x4')]
zs3 = [Symbol('x1'), Symbol('x2'),Symbol('x3')]
zs2 = [Symbol('x1'), Symbol('x2')]

alpha= Symbol('alpha')

class Newton2:
    def __init__(self,f,x0,tol,nv):
        self.nv=nv
        self.f=f
        self.x0=x0
        self.tol=tol
        self.i=0

    def initial_point(self):
        x0=self.x0
        i=1
        Xs=[]
        k=1
        while (i<len(x0)):
            if (x0[i]==','):
                nb=float(x0[k:i])
                Xs.append(nb)
                k=i+1
            i=i+1
        Xs.append(float(x0[k:len(x0)-1]))
        return np.array(Xs)
    def fp_methode(self,fa,a,b):
        def f(alpha):
            return eval(fa)

        # definir la derive de 1er ordre de la fonction utilise 
        def df(alpha):
            return derivative(f,alpha,0.01) # 0.01 est le pas choisi

        e=abs(a-b)
        xk_1=a
        xk=b
        while(e>=self.tol):
            if(df(xk_1)-df(xk)==0):
                return xk_1
            dk=-df(xk)*(xk_1-xk)/(df(xk_1)-df(xk))
            xk_Plus1=xk+dk
            e=abs(xk_Plus1-xk)
            xk_1=xk
            xk=xk_Plus1

        return xk_Plus1

    def newton_methode(self,fonc ,x0):
        t=self.tol
        def f(alpha):
            return eval(fonc)

        def df(alpha):
            return derivative(f,alpha,0.01)

        # definir la derive seconde de la fonction utilisee
        def d2f(alpha):
            return derivative(f,alpha,0.01,2)
        xk=x0
        while(True):
            if(d2f(xk)==0):
                xk_1=self.fp_methode(fonc,xk,xk+1)
                return xk_1
            xk_1=xk - (df(xk)/d2f(xk))
            k=xk
            xk=xk_1
            if (abs(xk_1-k)<t):
                break

        return xk_1

    def string2func(self):
        f=self.f
        ''' evaluates the string and returns a function of x '''
        # Trouver tous les mots dans la chaine de caracteres et verifier s'ils sont autorisés
        for word in re.findall('[a-zA-Z_]+', f):
            if word not in allowed_words:
                raise ValueError(
                    '"{}" is forbidden to use in math expression'.format(word)
                )

        # On remplace les mots trouves dans la chaine de caracteres
        for old, new in replacements.items():
            f = f.replace(old, new)

        # Eval(fonc) permet d'executer la chaine de caracteres fonc en tant qu' instruction Python
        return f

    # evaluer la fonction entrer comme string selon le nombre des variables entrées  
    def fonction1(self,x):

        if self.nv==2:
            f=str(eval(self.f, None, dict([z.name, z] for z in zs2)))
            f=f.replace('x1','(x1)')
            f=f.replace('x1','{x1}')
            f=f.replace('x2', '(x2)')
            f=f.replace('x2', '{x2}')
            f=f.format(x1=x[0],x2=x[1])
        elif(self.nv==3):
            f=str(eval(self.f, None, dict([z.name, z] for z in zs3)))
            f=f.replace('x1','(x1)')
            f=f.replace('x1','{x1}')
            f=f.replace('x2', '(x2)')
            f=f.replace('x2', '{x2}')
            f=f.replace('x3','(x3)')
            f=f.replace('x3','{x3}')
            f=f.format(x1=x[0],x2=x[1],x3=x[2])
        else:
            f=str(eval(self.f, None, dict([z.name, z] for z in zs4)))
            f=f.replace('x1','(x1)')
            f=f.replace('x1','{x1}')
            f=f.replace('x2', '(x2)')
            f=f.replace('x2', '{x2}')
            f=f.replace('x3','(x3)')
            f=f.replace('x3','{x3}')
            f=f.replace('x4','(x4)')
            f=f.replace('x4','{x4}')
            f=f.format(x1=x[0],x2=x[1],x3=x[2],x4=x[3])
        return np.float64(float(eval(f)))

    #la fonction d'evalution pour le 3d
    def fonction3d(self,x1,x2):
        f=self.string2func()
        return eval(f)

    #fonction pour reformuler la fonction entree en fonction de alpha  
    def alphaa(self,x_alpha,alpha):
        
        f=self.f
        f=f.replace('x1','(x1)')
        f=f.replace('x1','{x1}')
        f=f.replace('x2', '(x2)')
        f=f.replace('x2', '{x2}')
        f=f.format(x1=x_alpha[0],x2=x_alpha[1])
        if(self.nv==3):
            f=f.replace('x3','(x3)')
            f=f.replace('x3','{x3}')
            f=f.format(x3=x_alpha[2])
        if(self.nv==4):
            f=f.replace('x3','(x3)')
            f=f.replace('x3','{x3}')
            f=f.replace('x4','(x4)')
            f=f.replace('x4','{x4}')
            f=f.format(x3=x_alpha[2],x4=x_alpha[3])
        return f

    # la méthode qui permet de tester si l'hessien est définie positive ou non
    def is_pos_def(self,A):
        if np.allclose(A, A.T):
            try:
                L=np.linalg.cholesky(A)
                return 1
            except np.linalg.LinAlgError:
                return 0
        else:
            #print(2)
            return 0
    # l'algorithme de Newton en multidimentionnel
    def Newton2(self):
        dict=[]
        x=self.initial_point()
        alpha_k=0.2
        eps=random.uniform(0,1)
        # la matrice identité
        Ident=np.identity(self.nv)
        list=(self.i,str(x),self.fonction1(x),alpha_k)
        dict.insert(self.i,list)
        while(True):
            vals=[]# dictionnaire pour stocker les valeurs propres
            self.i+=1
            i=self.i
            # calculer le gradient de f pou un point x
            G=nd.Gradient(self.fonction1)(x)
            #calculer l'hessien de fpou un point x
            H=nd.Hessian(self.fonction1)(x)
            # retourner les valeurs propre et mes valeures propres de l'hessien
            w,v=eig(H)
            # stocker les valeures propre dans le dictionnaire
            for val in w:
                if val<=0:
                    val=0.1
                vals.append(val)

            # si l'hessien n'est pas définie positive, on modifie l'hessien en 
            #remplaçant les valeures propres négatives par un petit epsilon
            if self.is_pos_def(np.matrix(H))==0:
                print('non pos def!!!')
                W=np.array(vals)
                #une matrice diagonale des valeures propres
                dg=np.diag(W,0)
                l=np.dot(v,dg)
                H=np.dot(l,la.inv(v))
            
            x_alpha=x-alpha*np.matmul(la.inv(H), G)
            fc=self.alphaa(x_alpha,alpha)# la fonction en fonction de alpha
            alpha_k=self.newton_methode(fc ,alpha_k)
            alpha_k=abs(alpha_k)
            xk_PLUS_1=x-alpha_k*np.matmul(la.inv(H), G)
            if self.nv==2:# etudier les cas de nombre de variables entree par l'utilisateur
                xk_PLUS_1=np.array([float(xk_PLUS_1[0]),float(xk_PLUS_1[1])])
                xk_str="({0:.4f},{0:.4f})".format(xk_PLUS_1[0],xk_PLUS_1[1])
            elif self.nv==3:
                xk_PLUS_1=np.array([float(xk_PLUS_1[0]),float(xk_PLUS_1[1]),float(xk_PLUS_1[2])])
                xk_str="({0:.4f},{0:.4f})".format(xk_PLUS_1[0],xk_PLUS_1[1],xk_PLUS_1[2])
            elif self.nv==4:
                xk_PLUS_1=np.array([float(xk_PLUS_1[0]),float(xk_PLUS_1[1]),float(xk_PLUS_1[2]),float(xk_PLUS_1[3])])
                xk_str="({0:.4f},{0:.4f})".format(xk_PLUS_1[0],xk_PLUS_1[1],xk_PLUS_1[2],xk_PLUS_1[3])

            list=(i,xk_str,self.fonction1(xk_PLUS_1),alpha_k)
            dict.insert(i,list)
            if(la.norm(xk_PLUS_1-x)<self.tol or self.i==200):#condition d'arret
                break
            x=xk_PLUS_1

        return dict
   #la fonction de gradient par le pas exacte en 3d
    def GradientDescent3d(self):
        global fonction,func
        x=self.initial_point()
        # la matrice identité
        Ident=np.identity(self.nv)

        
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
        alpha_k=0.2
        while(True):
            G=nd.Gradient(self.fonction1)(x)
            H=nd.Hessian(self.fonction1)(x)
            w,v=eig(H)
            vals=[]
            for val in w:
                if val<=0:
                    val=0.1
                vals.append(val)
            if self.is_pos_def(np.matrix(H))==0:
                W=np.array(vals)
                dg=np.diag(W,0)
                l=np.dot(v,dg)
                H=np.dot(l,la.inv(v))
            

            x_alpha=x-alpha*np.matmul(la.inv(H), G)
            func=self.alphaa(x_alpha,alpha)
            alpha_k=self.newton_methode(func ,alpha_k)
            alpha_k=abs(alpha_k)
            #d_k=-np.matmul(la.inv(H),G)
            xk_PLUS_1=x-alpha_k*np.matmul(la.inv(H), G)

            xk_PLUS_1=np.array([float(xk_PLUS_1[0]),float(xk_PLUS_1[1])])
            x=xk_PLUS_1
            ax.scatter(xk_PLUS_1[0], xk_PLUS_1[1], self.fonction1(xk_PLUS_1), marker='o', s=10, color='#00FF00')
            plt.draw()
            plt.pause(0.05)
            #if(LA.norm(xk_PLUS_1-x)<self.tol):
                #break
            #x=xk_PLUS_1

