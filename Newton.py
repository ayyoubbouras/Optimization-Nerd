import sympy as smp
import numpy as np
from sympy import *
from math import *
from math import *


# la classe de l'algorithme de Newton
class NewtonMethod():
	# le constructeur de la classe qui prend comme paramètres f,x0 et tolérance
	def __init__(self,f,x0,N):
		self.f=f
		self.x0=x0
		self.N=N
	# La méthode qui permet de retourner l'image d'une valeur f(a)
	def func(self,a):
		x = smp.Symbol('x')#definir le caractère x comme une variable
		f = lambda x: float(eval(self.f))
		return f(a)
	# La méthode qui permet de calculer la dérivée de f pour un réel
	def Derivative_f(self,a):
		x = smp.Symbol('x')#definir le caractère x comme une variable
		deriv1= smp.diff(self.f,x)
		der1 = lambda x: float(eval(str(deriv1)))
		return der1(a)
	# La méthode qui permet de calculer la dérivée seconde de f pour un réel
	def Derivative2_f(self,a):
		x = smp.Symbol('x')#definir le caractère x comme une variable
		deriv1= smp.diff(self.f,x)
		deriv2=smp.diff(deriv1,x)
		der2 = lambda x: float(eval(str(deriv2)))
		return der2(a)

	# l'algorithme de newton
	def Newton(self):
		k = 1
		dict=[]#le dictionnaire où on stocke les itérations
		for i in range(self.N):
			# Si la dérivée seconde est nulle, un message d'erreur sera affiché
			if self.Derivative2_f(self.x0) == 0.0:
				mbox.showerror('Exception','Newton Raphson Methos is impossible in this case!\nSuggestion: You can choose False Position method ')
				break
			# la direction de Newton
			d=- float(self.Derivative_f(self.x0))/float(self.Derivative2_f(self.x0))
			#Mis à jour du point suivante
			x1 = self.x0 + d
			k = k + 1
			self.x0 = x1
			liste=(i,self.x0,self.func(self.x0),self.Derivative_f(self.x0),self.Derivative2_f(self.x0),d)
			dict.insert(i,liste)


		return dict







