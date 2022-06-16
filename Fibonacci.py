import sympy as smp
import numpy as np
from sympy import *
from math import *
from math import *
# la classe de l'algorithme de Fibanacci
class FibonacciMethod():
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
	# La fonction de Fibbonacci
	def fib(self,n):
		if n==0 or n==1:
			return 1
		else:
			return(self.fib(n-1)+self.fib(n-2))
	# l'algorithme de fibonacci
	def Fibonacci(self):
		#definir le caractère x comme une variable
		x = smp.Symbol('x')
		# la fonction f
		f = lambda x: float(eval(self.f))
		# la dérivée de f
		deriv1= smp.diff(self.f,x)
		d=abs(float(self.b)-float(self.a))
		N=1
		while self.fib(N)<= float(d/self.tol):
			N+=1
		dict=[]#le dictionnaire où on stocke les itérations
		for self.i in range(1,N):
			d=d*float((self.fib(N-self.i)/self.fib(N-self.i+1)))
			x1=self.b-d
			x2=self.a+d
			liste=(self.i,self.a,self.b,x1,x2,self.func(x1),self.func(x2),self.b-self.a)
			if self.func(x2)>=self.func(x1):
				self.b=x2
			else:
				self.a=x1
			dict.insert(self.i,liste)
		

		return dict







