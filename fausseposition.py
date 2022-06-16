import sympy as smp
import numpy as np
from sympy import *
from math import *
from math import *


# la classe de l'algorithme de Fibpnacci
class FaussePositionMethod():
    # le constructeur de la classe qui prend comme paramètres f,a,b et tolérance
    def __init__(self,f,a,b,tol):
        self.f=f
        self.a=a
        self.b=b
        self.tol=tol
        self.i=0
    # La méthode qui permet de retourner l'image d'une valeur f(a)
    def func(self,a):
        x = smp.Symbol('x')
        f = lambda x: float(eval(self.f))
        return f(a)
    # La méthode qui permet de calculer la dérivée de f pour un réel
    def Derivative_f(self,a):
        #definir le caractère x comme une variable
        x = smp.Symbol('x')
        # ladérivée de f
        deriv1= smp.diff(self.f,x)
        der1 = lambda x: float(eval(str(deriv1)))
        return der1(a)


    # l'algorithme de fausse position
    def fp_methode(self):
        dict=[]#le dictionnaire où on stocke les itérations
        i=self.i
        a=self.a
        b=self.b
        f_a=self.func(float(a))
        # le header de la table d'itérations
        liste=(i,a,f_a,abs(float(b)-float(a)))
        dict.insert(i,liste)
        e=abs(a-b)
        xk_1=a
        xk=b
        while(e>=self.tol):
            self.i+=1
            # la direction de la méthode de fausse position
            dk=-self.Derivative_f(xk)*(xk_1-xk)/(self.Derivative_f(xk_1)-self.Derivative_f(xk))
            xk_Plus1=xk+dk
            #| x_k+1 - x_k |
            e=abs(xk_Plus1-xk)
            i=self.i
            f_x=self.func(float(xk_Plus1))
            liste=(i,xk_Plus1,f_x,e)
            dict.insert(i,liste)   
            xk_1=xk
            xk=xk_Plus1
            
        return dict


