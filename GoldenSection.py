import sympy as smp
import numpy as np
from sympy import *
from math import *
from math import *

# la classe de l'algorithme de golden section
class GoldenSectionMethod():
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
	# La méthode qui permet de calculer la dérivée de f pour un réel
	def Derivative_f(self,a):
		#definir le caractère x comme une variable
		x = smp.Symbol('x')
		# la dérivé de f
		deriv1= smp.diff(self.f,x)
		der1 = lambda x: float(eval(str(deriv1)))
		return der1(a)
	# l'algorithme de golden section
	def GoldenSection(self):
		if self.a>=self.b:
			return (0)
		else:
			x = smp.Symbol('x')
			f = lambda x: float(eval(self.f))
			deriv1= smp.diff(self.f,x)
			d=abs(float(self.b)-float(self.a))
			dict=[]#le dictionnaire où on stocke les itérations
			head=['i','a','b','x1',"x2","f(x1)","f(x2)","b-a"]
			while self.b-self.a >=self.tol:
				d=d*float(0.618)
				x1=self.b-d
				x2=self.a+d
				if self.func(x2)>=self.func(x1):
					self.b=x2
				else:
					self.a=x1
				liste=(self.i,self.a,self.b,x1,x2,self.func(x1),self.func(x2),self.b-self.a)
				dict.insert(self.i,liste)
				self.i+=1
		

		return dict









