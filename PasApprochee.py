##############################################
#Importation des libreries necessaires pour la 3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
###############################################

###############################################
#Les outils mathmatiques
import math   
import numpy as np
from sympy import *
from scipy.misc import derivative
from numpy import linalg as LA
import numdifftools as nd
from sympy import Symbol
import re
#################################################

####################################################################
# declare le alpha et les variables de x1 a x4 comme des symboles
alpha= Symbol('alpha')
zs4 = [Symbol('x1'), Symbol('x2'),Symbol('x3'),Symbol('x4')]
zs3 = [Symbol('x1'), Symbol('x2'),Symbol('x3')]
zs2 = [Symbol('x1'), Symbol('x2')]
####################################################################

# On renomme les expressions saisies par l'utilisateur par des expressions deja definies par numpy
replacements = {
    'sin' : 'np.sin',
    'cos' : 'np.cos',
    'exp': 'np.exp',
    'sqrt': 'np.sqrt',
    '^': '**',
    'log':'np.log'
}
################################################################################

#############################################################################
# Les mots autorises dans la saisie de l'utilisateur
allowed_words = [
    'x',
    'sin',
    'cos',
    'sqrt',
    'exp',
    'log'
]

##############################################################################

#La classe du gradient par Armijo
class gradient_descentArmijo:
    def __init__(self,f,x0,tol,nv):
        self.nv=nv
        self.f=f
        self.x0=x0
        self.tol=tol
        self.i=0
    #la fonction pour transformer le point initial d'un string au numpy array
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
        l=x0[k:len(x0)-1]
        Xs.append(float(l))
        return np.array(Xs)
    # la fonction pour obtenir la direction du direction de descente sous forme des fleches
    def get_arrow(self,c,d):
        x = c[0]
        y = c[1]
        z = self.fonction1(c)
        u,v=nd.Gradient(self.fonction1)(c)
        w =0
        return x,y,z,-u,-v,w 
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
        elif self.nv==4:
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
    #evalution de la fonction pour le cas de 3d
    def fonction3d(self,x1,x2):
        f=self.string2func()
        return eval(f)
    # la fonction de armijo
    def armijo_rule(self,alpha_0,x,f_x,grad_x,d_x,c,beta): #d_x est la direction de descente d_x . grad_x <= 0
    # test f(x_new) \leq f(x_0) + c alpha ps{d_x}{grad_x}
        test = 1
        alpha = alpha_0
        while test: 
            x_new = x+alpha*d_x;
            if (self.fonction1(x_new)<=f_x+c*alpha*np.dot(grad_x,d_x)):
                test = 0
            else:
                alpha = alpha*beta
        return alpha         
    #la fonction de gradient qui appelle le pas de armijo
    def GradientDescent(self,c=0.1,L=100,beta=0.5):
        dict=[]
        x0=self.initial_point()
        x = x0
        x_old = x
        grad_x = nd.Gradient(self.fonction1)(x)
        d_x = -grad_x
        f_x = self.fonction1(x)
        alpha_0 = -(1./L)*np.dot(d_x,grad_x)/np.power(np.linalg.norm(d_x),2)
        h = self.armijo_rule(alpha_0,x,f_x,grad_x,d_x,c,beta)
        #alpha_k=0.2
        list=(self.i,str(x),self.fonction1(x),h)
        dict.insert(self.i,list)
        while(True):
            self.i+=1
            i=self.i
            x = x + h*d_x
            grad_x = nd.Gradient(self.fonction1)(x)
            f_x = self.fonction1(x)
            d_x = -grad_x
            alpha_0 = -(1./L)*np.dot(d_x,grad_x)/np.power(np.linalg.norm(d_x),2)
            h = self.armijo_rule(alpha_0,x,f_x,grad_x,d_x,c,beta)
            if self.nv==2:
                x=np.array([float(x[0]),float(x[1])])#,float(xk_PLUS_1[2]),float(xk_PLUS_1[3])])
                xk_str="({0:.4f},{0:.4f})".format(x[0],x[1])
            elif self.nv==3:
                x=np.array([float(x[0]),float(x[1]),float(x[2])])
                xk_str="({0:.4f},{0:.4f})".format(x[0],x[1],x[2])
            elif self.nv==4:
                x=np.array([float(x[0]),float(x[1]),float(x[2]),float(x[3])])
                xk_str="({0:.4f},{0:.4f})".format(x[0],x[1],x[2],x[3])
            list=(i,xk_str,self.fonction1(x),h)
            dict.insert(i,list)
            if(np.linalg.norm(x - x_old)<self.tol):
                break
            x_old = x 
        return dict
    #la fonction de gradient par armijo en 3d
    def GradientDescent3d(self,c=0.1,L=100,beta=0.5):
        global fonction,func,q
        x0=self.initial_point()
        d=-nd.Gradient(self.fonction1)(x0)
        
        fig=plt.figure()
        fig.set_size_inches(9, 7, forward=True)
        ax=Axes3D(fig, azim=-29, elev=49,auto_add_to_figure=False)
        fig.add_axes(ax)
        X = np.linspace(-3, 3, 30)
        Y = np.linspace(-3, 3, 30)
        Z = self.fonction3d(X[:,None],Y[None,:])
        X, Y=np.meshgrid(X, Y)
        ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
        q=ax.quiver(*(self.get_arrow(x0,d)), length=0.7, normalize=True)
        plt.xlabel("Paramètre 1 (x)")
        plt.ylabel("Paramètre 2 (y)")
        x = x0
        x_old = x
        grad_x = nd.Gradient(self.fonction1)(x)
        d_x = -grad_x
        f_x = self.fonction1(x)
        alpha_0 = -(1./L)*np.dot(d_x,grad_x)/np.power(np.linalg.norm(d_x),2)
        h = self.armijo_rule(alpha_0,x,f_x,grad_x,d_x,c,beta)
        while(True):
            self.i+=1
            i=self.i
            x = x + h*d_x
            grad_x = nd.Gradient(self.fonction1)(x)
            f_x = self.fonction1(x)
            d_x = -grad_x
            alpha_0 = -(1./L)*np.dot(d_x,grad_x)/np.power(np.linalg.norm(d_x),2)
            h = self.armijo_rule(alpha_0,x,f_x,grad_x,d_x,c,beta)
            x_old = x
            ax.scatter(x[0], x[1], self.fonction1(x), marker='o', s=10, color='#00FF00')
            if(i%10==0):
                ax.quiver(*(self.get_arrow(x,d_x)),length=0.7, normalize=True,color='#AA4A44')
            plt.draw()
            plt.pause(0.05)
#La classe du gradient par GoldStein
class gradient_descentGold:
    def __init__(self,f,x0,tol,nv):
        self.nv=nv
        self.f=f
        self.x0=x0
        self.tol=tol
        self.i=0
    #la fonction du point initial
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
    # evaluer la fonction entrer comme string selon le nombre des variables entrees
    def fonction1(self,x):
        global n
        nv=self.nv
        if nv==2:
            f=str(eval(self.f, None, dict([z.name, z] for z in zs2)))
            f=f.replace('x1','(x1)')
            f=f.replace('x1','{x1}')
            f=f.replace('x2', '(x2)')
            f=f.replace('x2', '{x2}')
            f=f.format(x1=x[0],x2=x[1])
        elif(nv==3):
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
        return float(eval(f))
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
        return f 
    # la fonction pour obtenir la direction du direction de descente   
    def get_arrow(self,c,d):
        x = c[0]
        y = c[1]
        z = self.fonction1(c)
        u,v=nd.Gradient(self.fonction1)(c)
        w = 0
        return x,y,z,-u,-v,w  
    # Trouver les intervales du pas alpha par goldstein 
    def find_interval(self,x0,dk):
        left = 0
        step = 1
        right = step + left
        k = 2
        while True:
            if self.fonction1(x0 + right * dk) < self.fonction1(x0 + left * dk):
                step *= k
                left = right
                right = left + step
                return left, right
            else:
                if right <= 0:
                    left = right
                    right = 0
                return left, right
    # chosir le alpha coresond pour chaque iteration
    def find_root(self,x0, dk, a, b):
        alpha1 = (a + b) / 2
        rho = 0.01
        t = 3
        xk = x0
        gk = nd.Gradient(self.fonction1)(xk)
        left = a
        right = b
        while True:
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
    #la fonction de gradient qui appelle le pas de armijo
    def GradientDescent(self):
        dict=[]
        x0=self.initial_point()
        xk = x0
        grad_x = nd.Gradient(self.fonction1)(xk)
        d_x = -grad_x
        f_x = self.fonction1(xk)
        list=(self.i,str(xk),f_x,0.1)
        dict.insert(self.i,list)
        while(np.linalg.norm(grad_x) > self.tol and self.i < 500):
            self.i+=1
            i=self.i
            grad_x = nd.Gradient(self.fonction1)(xk)
            f_x = self.fonction1(xk)
            d_x = -grad_x
            alpha1 = self.find_root(xk, d_x, self.find_interval(xk, d_x)[0], self.find_interval(xk, d_x)[1])  # trouver le alpha par golstein
            xk = xk + alpha1*d_x # iteration x_k+1
            if self.nv==2: # etudier les cas de nombre de variables entree par l'utilisateur
                x=np.array([float(xk[0]),float(xk[1])])
                xk_str="({0:.4f},{0:.4f})".format(x[0],x[1])
            elif self.nv==3:
                x=np.array([float(x[0]),float(x[1]),float(x[2])])
                xk_str="({0:.4f},{0:.4f})".format(xk[0],xk[1],xk[2])
            elif self.nv==4:
                x=np.array([float(x[0]),float(xk[1]),float(xk[2]),float(xk[3])])
                xk_str="({0:.4f},{0:.4f})".format(x[0],x[1],x[2],x[3])
            list=(i,xk_str,self.fonction1(xk),alpha1) # stocker les iterations sous forme liste
            dict.insert(i,list)
        return dict
    #la fonction de evalution pour le 3d
    def fonction3d(self,x1,x2):
        f=self.string2func()
        return eval(f)
    #la fonction de gradient par armijo en 3d
    def GradientDescent3d(self):
        xk=self.initial_point()
        d = -nd.Gradient(self.fonction1)(xk)
        fig=plt.figure()
        fig.set_size_inches(9, 7, forward=True)
        ax=Axes3D(fig, azim=-29, elev=49,auto_add_to_figure=False)
        fig.add_axes(ax)
        X = np.linspace(-3, 3, 30)
        Y = np.linspace(-3, 3, 30)
        Z = self.fonction3d(X[:,None],Y[None,:])
        X, Y=np.meshgrid(X, Y)
        ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
        ax.quiver(*(self.get_arrow(xk,d)))
        plt.xlabel("Paramètre 1 (x)")
        plt.ylabel("Paramètre 2 (y)")
        while(True):
            self.i+=1
            i=self.i
            grad_x = nd.Gradient(self.fonction1)(xk)
            f_x = self.fonction1(xk)
            d_x = -grad_x
            alpha1 = self.find_root(xk, d_x, self.find_interval(xk, d_x)[0], self.find_interval(xk, d_x)[1])
            xk = xk + alpha1*d_x
            ax.scatter(xk[0], xk[1], self.fonction1(xk), marker='o', s=10, color='#00FF00')
            ax.quiver(*(self.get_arrow(xk,d_x)))
            plt.draw()
            plt.pause(0.05)
            if(np.linalg.norm(d_x)<self.tol): # condition d'arret
                break



