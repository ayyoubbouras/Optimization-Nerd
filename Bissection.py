import sympy as smp
import numpy as np
from sympy import *
from math import *
from math import *


# la classe de l'algorithme de bissection
class BissectionMethod():
	# le constructeur de la classe qui prend comme paramètres f,a,b et tolérance
	def __init__(self,f,a,b,tol):
		self.f=f
		self.a=a
		self.b=b
		self.tol=tol
		self.i=0
	# La méthode qui permet de retourner l'image d'une valeur f(a)
	def func(self,a):
		#definir le caractère x comme une variable
		x = smp.Symbol('x')
		f = lambda x: float(eval(self.f))
		return f(a)

	# pour calculer la dérivée de f pour un réel
	def Derivative_f(self,a):
		x = smp.Symbol('x')
		deriv1= smp.diff(self.f,x)
		der1 = lambda x: float(eval(str(deriv1)))
		return der1(a)
		
	#pour calculer la dérivée seconde de f pour un réel
	def Derivative2_f(self,a):
		#definir le caractère x comme une variable
		x = smp.Symbol('x')
		deriv1= smp.diff(self.f,x)
		deriv2=smp.diff(deriv1,x)
		der2 = lambda x: float(eval(str(deriv2)))
		return der2(a)

	# l'algorithme de bissection
	def Bissection(self):
		if self.Derivative_f(self.a)*self.Derivative_f(self.b)>0:
			return (-1)
		elif self.a>=self.b:
			return (0)
		else:
			dict=[]#le dictionnaire où on stocke les itérations
			while abs(float(self.b)-float(self.a))>self.tol :
				c = float((float(self.a)+float(self.b))/2)
				derivee_c=self.Derivative_f(c)
				i=self.i
				a=self.a
				b=self.b
				f_a=self.func(float(a))
				f_b=self.func(float(b))
				f_c=self.func(float(c))
				liste=(i,a,b,c,f_a,f_b,f_c,float(derivee_c),abs(float(b)-float(a)))
				dict.insert(i,liste)
				if derivee_c<=0:
					if derivee_c==0 and self.Derivative2_f(c)>0:
						break
					else:
						self.a=c
				elif derivee_c>0:
					self.b=c
				
				
				self.i+=1


		return dict







