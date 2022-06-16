# Importation des libreries necessaires pour la fonctionnement du notre application

########################
#Les outils mathmatiques
import sympy as smp
import numpy as np
from sympy import *
from math import *
########################

###############################################################
#les packages et les modules utilises pour l'interface graphique
from tkinter import *
from tkinter import ttk
import interface_elements1 as IE
from tkinter import Menu  
from tkinter import messagebox as mbox 
##############################################################

###################################################
#les modules importes 
import Bissection as B
import fausseposition as fp
import GradientDescent as gd
import Newton as NW
import Newton2 as NW2
import Fibonacci as F
import GoldenSection as GS
import interface_elements1 as IE
import PasApprochee as PA
import QuasiNewton as QN
import Newton2 as NW2 
####################################################

###########################################################################
#Initialisation des variables utilises au sein de notre Interface Graphique
Entry_f='' # l'espace ou l'utilisateur va entree son fonction sous forme string
Entry_a=''  # le 'Entry_a' peut etre partie d'intervale [a,b] (bissection) entree par l'utilisateur ou juste le point initial ( newthon )
Entry_b='' # le 'Entry_b' est partie d'intervale [a,b] (bissection) 
Entry_tol='' # La tolerance entree par l'utilisateur
Input_f='' # cette variable va contenir la phrese "Entrer la fonction : "
Input_a=''  # cette variable va contenir la phrese "Entrer a : "
Input_b=''  # cette variable va contenir la phrese "Entrer b : "
Input_tol=''  # cette variable va contenir la phrese "Entrer la tolerance : " 
user_input='' # la totalite des input de utilisateur
page_biss='' # page de bissection
Bissection_page=''  # variable pour la page du Bissection
Fibonacci_page=''  # variable pour la page du Fibonacci
GoldenSection_page='' # variable pour la page du goldenSection
###########################################################################



#######################################################################
#all_children() :
   #est une fonction utilise pour realiser le boutton back qui nous aide pour le retour aux pages qui precede 
   #et il utilise pour selectionner chaque item de l'interface pour le supprimer apres apres la fonction back()  
def all_children():
	global Bissection_page
	_list=Bissection_page.winfo_children() 
	for item in _list: # loop pour selectionner chaque item
		if item.winfo_children(): 
			_list.extend(item.winfo_children())
	return _list

#######################################################################
#la fonction back() est appelee par le boutton back
#et elle sert a supprimer les items actuelles d'une page
def Back():
	global Bissection_page
	widget=all_children() # appel de la fonction all_children
	for item in widget: # loop qui supprime les items selectionne par all_children()
		item.destroy()
#######################################################################

# la page de Bissection en utilisant les outils de tkinter
def Bissection():
	global Entry_a,Entry_b,c,Input_a,Input_b,Entry_tol,Input_tol,user_input,Entry_f,Input_f,Bissection_page,root 
        # il est neccessaire de mettre les variable global pour qu'elles peuvent etre utilises au sein des fonctions
	Bissection_page=IE.create_page("Bissection Algorithm","#2f4f4f") # IE est le module interface_elements et la fonction create_page sert a cree une page au niveau de l'interface
	Bissection_page.tkraise()
	Title=Label(Bissection_page,text="Bissection Method",font="Helvectica 13",bg='#2f4f4f',fg='white')
	Title.place(x=650, y=15) # le titre de la page et son emplacement
	Input_f = Label(Bissection_page, text="Enter f(x):",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_f.place(x=450, y=50) 
	Entry_f= ttk.Entry(Bissection_page , font="Helvectica 12",width=30)
	Entry_f.place(x=600, y=50) #la case d'entree de la fonction
	Input_a = Label(Bissection_page, text="Enter a:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_a.place(x=450, y=90) 
	Entry_a= ttk.Entry(Bissection_page , font="Helvectica 12",width=30)
	Entry_a.place(x=600, y=90) #la case d'entree de 'a'
	Input_b = Label(Bissection_page, text="Enter b (b>a):",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_b.place(x=450, y=130) 
	Entry_b = ttk.Entry(Bissection_page , font="Helvectica 12",width=30)
	Entry_b.place(x=600, y=130) #la case d'entree de 'b'
	Input_tol = Label(Bissection_page, text="Enter Tolerance:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_tol.place(x=450, y=170) 
	Entry_tol = ttk.Entry(Bissection_page,font="Helvectica 12",width=30)
	Entry_tol.place(x=600, y=170) #la case d'entree de la tolerance
	btn1 = Button(Bissection_page, text = 'Back',bg="#5f9ea0",fg="white",font="Helvectica 9",width="9",command=root)
	btn1.place(x=600, y=210) # Le boutton back et son emplacement
	btn2 = Button(Bissection_page, text = 'Apply',bg="#5f9ea0",font="Helvectica 9",fg="white",width="9",command=showResultBissection)
	btn2.place(x=805, y=210) #Le boutton Apply et son emplacement
	graph=Label(Bissection_page,text="Graph of f(x) & f'(x)",bg="white",fg='#2f4f4f',font="Courier 11")
	graph.place(x=915,y=370) #L'espace consacre pour les graphes
	iterations=Label(Bissection_page,text="Iterations table",bg="white",fg='#2f4f4f',font="Courier 11")
	iterations.place(x=280,y=370) #L'espace consacre pour les iterations
	Bissection_page.mainloop()


# la fonction qui nous aide pour appeler l'algorithme de bissection et storer son resulat pour l'afficher en forme d'iterations et graphe
def showResultBissection():
	global Entry_a,Entry_b,c,Input_a,Input_b,Entry_tol,Input_tol,user_input,Entry_f,Input_f,Bissection_page
        # il est neccessaire de mettre les variable global pour qu'elles peuvent etre utilises au sein des fonctions
	user_input=Entry_f.get() #recevoir la fonction entree par l'utilisateur et le stocker dans la variable user_input
	if (len(Entry_f.get() )==0):#cas d'exeption quand la fonction n'est pas saisie
		mbox.showinfo('Error!','Function entry is empty! Please input a function!')
	else:
		try:
			a=float(Entry_a.get()) #recevoir a de l'intervalle [a,b] et le stocker dans une variable
			b=float(Entry_b.get())  #recevoir b de l'intervalle [a,b]  et le stocker dans une variable
			tol=float(Entry_tol.get()) #recevoir la Tolerance entree par l'utilisateur 
			animation=[] #table pour stocker les iterations
			animation.append(a) #ajouter a au tableau animation
			animation.append(b) #ajouter b au tableau animation
			biss=B.BissectionMethod(user_input,a,b,tol) # appel et intialisation de la classe Bissection (biss est un objet de la classe)
			m=a
			k=b
			x=smp.Symbol('x') # declarer 'x' comme un symbole
			deriv1= smp.diff(user_input,x) #calculer la derivee par rapport a 'x'
			res=biss.Bissection() # res contient les iterations en forme de listes

			if res==-1: #exception : il n'existe pas de minimum
				mbox.showerror('Exception',"There is no minimum in the chosen interval!!\n Pleas choose an other interval ( f'(a)*f'(b)<=0 )")
			elif res==0: #exception l'interval est incorrecte
				mbox.showerror('Exception',"Invalid interval !\nNB: a must be bellow to b")
			else:
				for anim in res: #loop pour extraire les iterations du resultat 'res' et les stocker dans le tableau animation puis le transformer en numpy array .
					animation.append(anim[1])
					xtab= np.array(animation)
				n=biss.i # stocker le nombre des iterations dans la variables n
				head=['i','a','b','c',"f(a)","f(b)","f(c)","f'(c)","|b-a|"]
				result=IE.table(Bissection_page,head,60)
				for i in range(0,n): #creation de la tables des iterations
					liste=res[i]
					result.insert(parent='',index=i,iid=i,text='',values=liste)
					i+=1
				result.place(x=115,y=450) #Emplacement de la table des iterations
				IE.plot(user_input,m-2,k+2,Bissection_page,str(deriv1),xtab) #Plotting des graphes

		except Exception as e: #exception la nature des entrees n'est pas valide
			print(e) 
			mbox.showerror('Exception!','Missing or invalid inputs \nNB : a, b, & Tolerance must be a numbers\n      Tolerance must be positif')

#################################################################################################################
# la page de Fibonacci en utilisant les outils de tkinter
def Fibonacci():
	global Entry_a,Entry_b,c,Input_a,Input_b,Entry_tol,Input_tol,user_input,Entry_f,Input_f,Fibonacci_page,root
        # il est neccessaire de mettre les variable global pour qu'elles peuvent etre utilises au sein des fonctions
	Fibonacci_page=IE.create_page("Fibonacci Algorithm","#2f4f4f") # IE est le module interface_elements et la fonction create_page sert a cree une page au niveau de l'interface
	Fibonacci_page.tkraise()
	Title=Label(Fibonacci_page,text="Fibonacci Method",font="Helvectica 13",bg='#2f4f4f',fg='white')
	Title.place(x=650, y=15) # le titre de la page et son emplacement
	Input_f = Label(Fibonacci_page, text="Enter f(x):",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_f.place(x=450, y=50) #la case d'entree de la fonction
	Entry_f= ttk.Entry(Fibonacci_page , font="Helvectica 12",width=30)
	Entry_f.place(x=600, y=50)
	Input_a = Label(Fibonacci_page, text="Enter a:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_a.place(x=450, y=90) #la case d'entree de 'a'
	Entry_a= ttk.Entry(Fibonacci_page , font="Helvectica 12",width=30)
	Entry_a.place(x=600, y=90)
	Input_b = Label(Fibonacci_page, text="Enter b (b>a):",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_b.place(x=450, y=130) #la case d'entree de 'b'
	Entry_b = ttk.Entry(Fibonacci_page , font="Helvectica 12",width=30)
	Entry_b.place(x=600, y=130)
	Input_tol = Label(Fibonacci_page, text="Enter Tolerance:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_tol.place(x=450, y=170)  #la case d'entree de la tolerance
	Entry_tol = ttk.Entry(Fibonacci_page,font="Helvectica 12",width=30)
	Entry_tol.place(x=600, y=170)
	btn1 = Button(Fibonacci_page, text = 'Back',bg="#5f9ea0",fg="white",font="Helvectica 9",width="9",command=root.tkraise)
	btn1.place(x=600, y=210)  # Le boutton back et son emplacement
	btn2 = Button(Fibonacci_page, text = 'Apply',bg="#5f9ea0",font="Helvectica 9",fg="white",width="9",command=showResultFibonacci)
	btn2.place(x=805, y=210) #Le boutton Apply et son emplacement
	graph=Label(Fibonacci_page,text="Graph of f(x) & f'(x)",bg="white",fg='#2f4f4f',font="Courier 11")
	graph.place(x=915,y=370)  #L'espace consacre pour les graphes
	iterations=Label(Fibonacci_page,text="Iterations table",bg="white",fg='#2f4f4f',font="Courier 11")
	iterations.place(x=280,y=370)
	Fibonacci_page.mainloop()


# la fonction qui nous aide pour appeler l'algorithme de Fibonacci et storer son resulat pour l'afficher en forme d'iteration et graphe
def showResultFibonacci():
	global Entry_a,Entry_b,c,Input_a,Input_b,Entry_tol,Input_tol,user_input,Entry_f,Input_f,Fibonacci_page
        # il est neccessaire de mettre les variable global pour qu'elles peuvent etre utilises au sein des fonctions
	user_input=Entry_f.get() #recevoir la fonction entree par l'utilisateur et le stocker dans la variable user_input
	if (len(Entry_f.get() )==0): #cas d'exeption quand la fonction n'est pas saisie
		mbox.showinfo('Error!','Function entry is empty! Pleas input a function!')
	else:
		try:
			a=float(Entry_a.get()) #recevoir a de l'intervalle [a,b] et le stocker dans une variable
			b=float(Entry_b.get()) #recevoir b de l'intervalle [a,b] et le stocker dans une variable
			tol=float(Entry_tol.get()) #recevoir la Tolerance entree par l'utilisateur
			animation=[]  #table pour stocker les iterations
			animation.append(a) #ajouter a au tableau animation
			animation.append(b)  #ajouter a au tableau animation
			fib=F.FibonacciMethod(user_input,a,b,tol) # appel et intialisation de la classe Fibonacci (fib est un objet de la classe)
			m=a
			k=b
			x=smp.Symbol('x') # declarer 'x' comme un symbole
			deriv1= smp.diff(user_input,x) #calculer la derivee par rapport a 'x'
			res=fib.Fibonacci() # res contient les iterations en forme de listes
			if res==-1: #exception : il n'existe pas de minimum
				mbox.showerror('Exception',"There is no minimum in the chosen interval!!\n Pleas choose an other interval ( f'(a)*f'(b)<=0 )")
			elif res==0: #exception : l'intervalle est incorrecte
				mbox.showerror('Exception',"Invalid interval !\nNB: a must be bellow to b")
			else:
				for anim in res:
					animation.append(anim[1])
					xtab= np.array(animation)
				n=fib.i
				#print(n)
				head=['i','a','b','x1',"x2","f(x1)","f(x2)","b-a"]
				result=IE.table(Fibonacci_page,head,60)
				for i in range(0,n):
					liste=res[i]
					result.insert(parent='',index=i,iid=i,text='',values=liste)
					i+=1
				result.place(x=115,y=450)
				IE.plot(user_input,m-2,k+2,Fibonacci_page,str(deriv1),xtab)  #Plotting des graphes

		except Exception as e:  #exception la nature des entrees n'est pas valide
			print(e)
			mbox.showerror('Exception!','Missing or invalid inputs \nNB : a, b, & Tolerance must be a numbers\n      Tolerance must be positif')
#################################################################################################################
# la page de Fausse Position en utilisant les outils de tkinter
def FaussePosition():
	global Entry_a,Entry_b,c,Input_a,Input_b,Entry_tol,Input_tol,user_input,Entry_f,Input_f,FaussePosition_page,root
	FaussePosition_page=IE.create_page("Fausse Position Algorithm","#2f4f4f")
	FaussePosition_page.tkraise()
	Title=Label(FaussePosition_page,text="Fausse Position Method",font="Helvectica 13",bg='#2f4f4f',fg='white')
	Title.place(x=650, y=15)
	Input_f = Label(FaussePosition_page, text="Enter f(x):",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_f.place(x=450, y=50) 
	Entry_f= ttk.Entry(FaussePosition_page , font="Helvectica 12",width=30)
	Entry_f.place(x=600, y=50)
	Input_a = Label(FaussePosition_page, text="Enter x0:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_a.place(x=450, y=90) 
	Entry_a= ttk.Entry(FaussePosition_page , font="Helvectica 12",width=30)
	Entry_a.place(x=600, y=90)
	Input_b = Label(FaussePosition_page, text="Enter x1:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_b.place(x=450, y=130) 
	Entry_b = ttk.Entry(FaussePosition_page, font="Helvectica 12",width=30)
	Entry_b.place(x=600, y=130)
	Input_tol = Label(FaussePosition_page, text="Enter Tolerance:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_tol.place(x=450, y=170) 
	Entry_tol = ttk.Entry(FaussePosition_page,font="Helvectica 12",width=30)
	Entry_tol.place(x=600, y=170)
	btn1 = Button(FaussePosition_page, text = 'Back',bg="#5f9ea0",fg="white",font="Helvectica 9",width="9",command=root.tkraise)
	btn1.place(x=600, y=210)
	btn2 = Button(FaussePosition_page, text = 'Apply',bg="#5f9ea0",font="Helvectica 9",fg="white",width="9",command=showResultFP)
	btn2.place(x=805, y=210)
	graph=Label(FaussePosition_page,text="Graph of f(x) & f'(x)",bg="white",fg='#2f4f4f',font="Courier 11")
	graph.place(x=915,y=370)
	iterations=Label(FaussePosition_page,text="Iterations table",bg="white",fg='#2f4f4f',font="Courier 11")
	iterations.place(x=280,y=370)
	FaussePosition_page.mainloop()


# la fonction qui nous aide pour appeler l'algorithme de fausse position et storer son resulat pour l'afficher en forme d'iteration et graphe
def showResultFP():
    global Entry_a,Entry_b,Input_a,Input_b,Entry_tol,Input_tol,user_input,Entry_f,Input_f,FaussePosition_page
    user_input=Entry_f.get()
    if (len(Entry_f.get() )==0):
        mbox.showinfo('Error!','Function entry is empty! Please input a function!')
    else:
        a=float(Entry_a.get())
        b=float(Entry_b.get())
        tol=float(Entry_tol.get())
        animation=[]
        animation.append(a)
        animation.append(b)
        fausse=fp.FaussePositionMethod(user_input,a,b,tol) # appel et intialisation de la classe Fausse Position (fausse est un objet de la classe)
        m=a
        k=b
        x=smp.Symbol('x')
        deriv1= smp.diff(user_input,x)
        res=fausse.fp_methode()
        for anim in res:
        	animation.append(anim[1])
        xtab= np.array(animation)
        n=fausse.i
        head=['i','x_k',"f(x_k)","|x_k+1-x_k|"]
        result=IE.table(FaussePosition_page,head,60)
        for i in range(0,n):
            liste=res[i]
            result.insert(parent='',index=i,iid=i,text='',values=liste)
        result.place(x=115,y=450)
        IE.plot(user_input,m-2,m+10,FaussePosition_page,str(deriv1),xtab)

#################################################################################################################
# la page de Gradient en utilisant les outils de tkinter
def Grad_Descent():
	global Entry_a,c,Input_a,Entry_tol,Input_tol,user_input,Entry_f,Input_f,Entry_n,Input_n,GradientDescent_page,root
	GradientDescent_page=IE.create_page("Gradient Descent Algorithm","#2f4f4f")
	GradientDescent_page.tkraise()
	Title=Label(GradientDescent_page,text="Gradient Descent Method",font="Helvectica 13",bg='#2f4f4f',fg='white')
	Title.place(x=650, y=15)
	Input_n = Label(GradientDescent_page, text="Enter number of variables:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_n.place(x=420, y=50) 
	Entry_n= ttk.Entry(GradientDescent_page , font="Helvectica 12",width=30)
	Entry_n.place(x=600, y=50)
	Input_f = Label(GradientDescent_page, text="Enter f(x1,x2,..):",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_f.place(x=450, y=90) 
	Entry_f= ttk.Entry(GradientDescent_page , font="Helvectica 12",width=30)
	Entry_f.place(x=600, y=90)
	Input_a = Label(GradientDescent_page, text="Enter x0:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_a.place(x=450, y=130) 
	Entry_a= ttk.Entry(GradientDescent_page , font="Helvectica 12",width=30)
	Entry_a.place(x=600, y=130)
	Input_tol = Label(GradientDescent_page, text="Enter Tolerance:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_tol.place(x=450, y=170) 
	Entry_tol = ttk.Entry(GradientDescent_page,font="Helvectica 12",width=30)
	Entry_tol.place(x=600, y=170)
	btn1 = Button(GradientDescent_page, text = 'Back',bg="#5f9ea0",fg="white",font="Helvectica 9",width="9",command=root.tkraise)
	btn1.place(x=600, y=210)
	btn2 = Button(GradientDescent_page, text = 'Apply',bg="#5f9ea0",font="Helvectica 9",fg="white",width="9",command=showResultGradDescent)
	btn2.place(x=805, y=210)
	graph=Label(GradientDescent_page,text="Graph of f(x) & f'(x)",bg="white",fg='#2f4f4f',font="Courier 11")
	graph.place(x=915,y=370)
	iterations=Label(GradientDescent_page,text="Iterations table",bg="white",fg='#2f4f4f',font="Courier 11")
	iterations.place(x=280,y=370)
	GradientDescent_page.mainloop()


# la fonction qui nous aide pour appeler l'algorithme de gradient et storer son resulat pour l'afficher en forme d'iteration et 3d
def showResultGradDescent():
    global Entry_a,Input_a,Entry_tol,Input_tol,Entry_n,Input_n,user_input,Entry_f,Input_f,GradientDescent_page
    user_input=Entry_f.get()
    if (len(Entry_f.get() )==0):
        mbox.showinfo('Error!','Function entry is empty! Please enter a function!')
    else:
        a=Entry_a.get()
        tol=float(Entry_tol.get())
        nv=int(Entry_n.get())

        gradDescent=gd.gradient_descent(user_input,a,tol,nv) # appel et intialisation de la classe du Gradient (gradDescent est un objet de la classe)
        res=gradDescent.GradientDescent()
        n=gradDescent.i # le nombre des iterations
        head=['i','x_k',"f(x_k)","Le Pas"]
        result=IE.table(GradientDescent_page,head,150)
        for i in range(0,n):
            liste=res[i]
            result.insert(parent='',index=i,iid=i,text='',values=liste)
        result.place(x=115,y=450)
        #IE.plot(user_input,m-2,m+10,FaussePosition_page,str(deriv1),xtab)
        if nv==2: # si l'utilisateur a chosit deux variables on applle la fonction du 3d
        	gradDescent.GradientDescent3d()

#################################################################################################################
#################################################################################################################
# la page de Newton Multidimentionnel en utilisant les outils de tkinter
def Newton2():
	global Entry_a,c,Input_a,Entry_tol,Input_tol,user_input,Entry_f,Input_f,Entry_n,Input_n,Newton2_page,root
	Newton2_page=IE.create_page("Newton Algorithm","#2f4f4f")
	Newton2_page.tkraise()
	Title=Label(Newton2_page,text="Newton Method",font="Helvectica 13",bg='#2f4f4f',fg='white')
	Title.place(x=650, y=15)
	Input_n = Label(Newton2_page, text="Enter number of variables:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_n.place(x=420, y=50) 
	Entry_n= ttk.Entry(Newton2_page , font="Helvectica 12",width=30)
	Entry_n.place(x=600, y=50)
	Input_f = Label(Newton2_page, text="Enter f(x1,x2,..):",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_f.place(x=450, y=90) 
	Entry_f= ttk.Entry(Newton2_page , font="Helvectica 12",width=30)
	Entry_f.place(x=600, y=90)
	Input_a = Label(Newton2_page, text="Enter x0:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_a.place(x=450, y=130) 
	Entry_a= ttk.Entry(Newton2_page , font="Helvectica 12",width=30)
	Entry_a.place(x=600, y=130)
	Input_tol = Label(Newton2_page, text="Enter Tolerance:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_tol.place(x=450, y=170) 
	Entry_tol = ttk.Entry(Newton2_page,font="Helvectica 12",width=30)
	Entry_tol.place(x=600, y=170)
	btn1 = Button(Newton2_page, text = 'Back',bg="#5f9ea0",fg="white",font="Helvectica 9",width="9",command=root.tkraise)
	btn1.place(x=600, y=210)
	btn2 = Button(Newton2_page, text = 'Apply',bg="#5f9ea0",font="Helvectica 9",fg="white",width="9",command=showResultNewton2)
	btn2.place(x=805, y=210)
	graph=Label(Newton2_page,text="Graph of f(x) & f'(x)",bg="white",fg='#2f4f4f',font="Courier 11")
	graph.place(x=915,y=370)
	iterations=Label(Newton2_page,text="Iterations table",bg="white",fg='#2f4f4f',font="Courier 11")
	iterations.place(x=280,y=370)
	Newton2_page.mainloop()


# la fonction qui nous aide pour appeler l'algorithme de newton multidim et storer son resulat pour l'afficher en forme d'iteration et 3d
def showResultNewton2():
    global Entry_a,Input_a,Entry_tol,Input_tol,Entry_n,Input_n,user_input,Entry_f,Input_f,Newton2_page
    user_input=Entry_f.get()
    if (len(Entry_f.get() )==0):
        mbox.showinfo('Error!','Function entry is empty! Please enter a function!')
    else:
        a=Entry_a.get()
        tol=float(Entry_tol.get())
        nv=int(Entry_n.get())
        newton2=NW2.Newton2(user_input,a,tol,nv)
        #x=smp.Symbol('x')
        #deriv1= smp.diff(user_input,x)
        res=newton2.Newton2()
        n=newton2.i
        head=['i','x_k',"f(x_k)","Le Pas"]
        result=IE.table(Newton2_page,head,150)
        for i in range(0,n):
            liste=res[i]
            result.insert(parent='',index=i,iid=i,text='',values=liste)
        result.place(x=115,y=450)
        #IE.plot(user_input,m-2,m+10,FaussePosition_page,str(deriv1),xtab)
        if nv==2:
        	newton2.GradientDescent3d()
#################################################################################################################
# la page de Gradient par le pas de Armijo en utilisant les outils de tkinter
def GradArmijo():
	global Entry_a,c,Input_a,Entry_tol,Input_tol,user_input,Entry_f,Input_f,Entry_n,Input_n,GradientDescentAr_page,root
	GradientDescentAr_page=IE.create_page("Gradient Descent Algorithm Par Armijo","#2f4f4f")
	GradientDescentAr_page.tkraise()
	Title=Label(GradientDescentAr_page,text="Gradient Descent Method Par Armijo",font="Helvectica 13",bg='#2f4f4f',fg='white')
	Title.place(x=650, y=15)
	Input_n = Label(GradientDescentAr_page, text="Enter number of variables:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_n.place(x=420, y=50) 
	Entry_n= ttk.Entry(GradientDescentAr_page , font="Helvectica 12",width=30)
	Entry_n.place(x=600, y=50)
	Input_f = Label(GradientDescentAr_page, text="Enter f(x1,x2,..):",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_f.place(x=450, y=90) 
	Entry_f= ttk.Entry(GradientDescentAr_page , font="Helvectica 12",width=30)
	Entry_f.place(x=600, y=90)
	Input_a = Label(GradientDescentAr_page, text="Enter x0:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_a.place(x=450, y=130) 
	Entry_a= ttk.Entry(GradientDescentAr_page , font="Helvectica 12",width=30)
	Entry_a.place(x=600, y=130)
	Input_tol = Label(GradientDescentAr_page, text="Enter Tolerance:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_tol.place(x=450, y=170) 
	Entry_tol = ttk.Entry(GradientDescentAr_page,font="Helvectica 12",width=30)
	Entry_tol.place(x=600, y=170)
	btn1 = Button(GradientDescentAr_page, text = 'Back',bg="#5f9ea0",fg="white",font="Helvectica 9",width="9",command=root.tkraise)
	btn1.place(x=600, y=210)
	btn2 = Button(GradientDescentAr_page, text = 'Apply',bg="#5f9ea0",font="Helvectica 9",fg="white",width="9",command=showResultGradArmijo)
	btn2.place(x=805, y=210)
	graph=Label(GradientDescentAr_page,text="Graph of f(x) & f'(x)",bg="white",fg='#2f4f4f',font="Courier 11")
	graph.place(x=915,y=370)
	iterations=Label(GradientDescentAr_page,text="Iterations table",bg="white",fg='#2f4f4f',font="Courier 11")
	iterations.place(x=280,y=370)
	GradientDescentAr_page.mainloop()


# la fonction qui nous aide pour appeler l'algorithme de gradient par le pad d'armijo et storer son resulat pour l'afficher en forme d'iteration et 3d
def showResultGradArmijo():
    global Entry_a,Input_a,Entry_tol,Input_tol,user_input,Entry_f,Input_f,Entry_n,Input_n,GradientDescentAr_page
    user_input=Entry_f.get()
    if (len(Entry_f.get() )==0):
        mbox.showinfo('Error!','Function entry is empty! Please enter a function!')
    else:
        nv=int(Entry_n.get())
        a=Entry_a.get()
        tol=float(Entry_tol.get())
        gradDescentAr=PA.gradient_descentArmijo(user_input,a,tol,nv)
        res=gradDescentAr.GradientDescent()
        n=gradDescentAr.i
        head=['i','x_k',"f(x_k)","Le Pas"]
        result=IE.table(GradientDescentAr_page,head,150)
        for i in range(0,n):
            liste=res[i]
            result.insert(parent='',index=i,iid=i,text='',values=liste)
        result.place(x=115,y=450)
        #IE.plot(user_input,m-2,m+10,FaussePosition_page,str(deriv1),xtab)
        if nv==2:
        	gradDescentAr.GradientDescent3d()

#################################################################################################################
#################################################################################################################
# la page de Gradient par le pas de Goldstein en utilisant les outils de tkinter
def GradGold():
	global Entry_a,c,Input_a,Entry_tol,Input_tol,user_input,Entry_f,Input_f,Entry_n,Input_n,GradientDescentgold_page,root
	GradientDescentgold_page=IE.create_page("Gradient Descent Algorithm Par GoldStein","#2f4f4f")
	GradientDescentgold_page.tkraise()
	Title=Label(GradientDescentgold_page,text="Gradient Descent Method Par Pas Approche GoldStein",font="Helvectica 13",bg='#2f4f4f',fg='white')
	Title.place(x=650, y=15)
	Input_n = Label(GradientDescentgold_page, text="Enter number of variables:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_n.place(x=420, y=50) 
	Entry_n= ttk.Entry(GradientDescentgold_page , font="Helvectica 12",width=30)
	Entry_n.place(x=600, y=50)
	Input_f = Label(GradientDescentgold_page, text="Enter f(x1,x2,..):",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_f.place(x=450, y=90) 
	Entry_f= ttk.Entry(GradientDescentgold_page , font="Helvectica 12",width=30)
	Entry_f.place(x=600, y=90)
	Input_a = Label(GradientDescentgold_page, text="Enter x0:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_a.place(x=450, y=130) 
	Entry_a= ttk.Entry(GradientDescentgold_page , font="Helvectica 12",width=30)
	Entry_a.place(x=600, y=130)
	Input_tol = Label(GradientDescentgold_page, text="Enter Tolerance:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_tol.place(x=450, y=170) 
	Entry_tol = ttk.Entry(GradientDescentgold_page,font="Helvectica 12",width=30)
	Entry_tol.place(x=600, y=170)
	btn1 = Button(GradientDescentgold_page, text = 'Back',bg="#5f9ea0",fg="white",font="Helvectica 9",width="9",command=root.tkraise)
	btn1.place(x=600, y=210)
	btn2 = Button(GradientDescentgold_page, text = 'Apply',bg="#5f9ea0",font="Helvectica 9",fg="white",width="9",command=showResultGradGold)
	btn2.place(x=805, y=210)
	graph=Label(GradientDescentgold_page,text="Graph of f(x) & f'(x)",bg="white",fg='#2f4f4f',font="Courier 11")
	graph.place(x=915,y=370)
	iterations=Label(GradientDescentgold_page,text="Iterations table",bg="white",fg='#2f4f4f',font="Courier 11")
	iterations.place(x=280,y=370)
	GradientDescentgold_page.mainloop()


# la fonction qui nous aide pour appeler l'algorithme de gradient par pas de goldstein et storer son resulat pour l'afficher en forme d'iteration et 3d
def showResultGradGold():
    global Entry_a,Input_a,Entry_tol,Input_tol,user_input,Entry_f,Input_f,Entry_n,Input_n,GradientDescentgold_page
    user_input=Entry_f.get()
    if (len(Entry_f.get() )==0):
        mbox.showinfo('Error!','Function entry is empty! Please enter a function!')
    else:
        nv=int(Entry_n.get())
        a=Entry_a.get()
        tol=float(Entry_tol.get())
        gradDescentGold=PA.gradient_descentGold(user_input,a,tol,nv)
        res=gradDescentGold.GradientDescent()
        n=gradDescentGold.i
        head=['i','x_k',"f(x_k)","Le Pas"]

        result=IE.table(GradientDescentgold_page,head,150)
        for i in range(0,n):
            liste=res[i]
            result.insert(parent='',index=i,iid=i,text='',values=liste)
        result.place(x=115,y=450)
        #IE.plot(user_input,m-2,m+10,FaussePosition_page,str(deriv1),xtab)
        if nv==2:
        	gradDescentGold.GradientDescent3d()

#################################################################################################################
# la page de goldensection en utilisant les outils de tkinter
def GoldenSection():
	global Entry_a,Entry_b,c,Input_a,Input_b,Entry_tol,Input_tol,user_input,Entry_f,Input_f,GoldenSection_page,root
	GoldenSection_page=IE.create_page("Golden Section Algorithm","#2f4f4f")
	GoldenSection_page.tkraise()
	Title=Label(GoldenSection_page,text="GoldenSection Method",font="Helvectica 13",bg='#2f4f4f',fg='white')
	Title.place(x=650, y=15)
	Input_f = Label(GoldenSection_page, text="Enter f(x):",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_f.place(x=450, y=50) 
	Entry_f= ttk.Entry(GoldenSection_page , font="Helvectica 12",width=30)
	Entry_f.place(x=600, y=50)
	Input_a = Label(GoldenSection_page, text="Enter a:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_a.place(x=450, y=90) 
	Entry_a= ttk.Entry(GoldenSection_page , font="Helvectica 12",width=30)
	Entry_a.place(x=600, y=90)
	Input_b = Label(GoldenSection_page, text="Enter b (b>a):",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_b.place(x=450, y=130) 
	Entry_b = ttk.Entry(GoldenSection_page , font="Helvectica 12",width=30)
	Entry_b.place(x=600, y=130)
	Input_tol = Label(GoldenSection_page, text="Enter Tolerance:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_tol.place(x=450, y=170) 
	Entry_tol = ttk.Entry(GoldenSection_page,font="Helvectica 12",width=30)
	Entry_tol.place(x=600, y=170)
	btn1 = Button(GoldenSection_page, text = 'Back',bg="#5f9ea0",fg="white",font="Helvectica 9",width="9",command=root.tkraise)
	btn1.place(x=600, y=210)
	btn2 = Button(GoldenSection_page, text = 'Apply',bg="#5f9ea0",font="Helvectica 9",fg="white",width="9",command=showResultGoldenSection)
	btn2.place(x=805, y=210)
	graph=Label(GoldenSection_page,text="Graph of f(x) & f'(x)",bg="white",fg='#2f4f4f',font="Courier 11")
	graph.place(x=915,y=370)
	iterations=Label(GoldenSection_page,text="Iterations table",bg="white",fg='#2f4f4f',font="Courier 11")
	iterations.place(x=280,y=370)
	GoldenSection_page.mainloop()


# la fonction qui nous aide pour appeler l'algorithme de gradient par pas de goldstein et storer son resulat pour l'afficher en forme d'iteration et graphe
def showResultGoldenSection():
	global Entry_a,Entry_b,c,Input_a,Input_b,Entry_tol,Input_tol,user_input,Entry_f,Input_f,GoldenSection_page
	user_input=Entry_f.get()
	if (len(Entry_f.get() )==0):
		mbox.showinfo('Error!','Function entry is empty! Pleas input a function!')
	else:
		try:
			a=float(Entry_a.get())
			b=float(Entry_b.get())
			tol=float(Entry_tol.get())
			animation=[]
			animation.append(a)
			animation.append(b)
			Golden=GS.GoldenSectionMethod(user_input,a,b,tol)
			m=a
			k=b
			x=smp.Symbol('x')
			deriv1= smp.diff(user_input,x)
			res=Golden.GoldenSection()
			if res==0:
				mbox.showerror('Exception',"Invalid interval !\nNB: a must be bellow to b")
			else:
				for anim in res:
					animation.append(anim[1])
					xtab= np.array(animation)
				n=Golden.i
				#print(n)
				head=['i','a','b','x1',"x2","f(x1)","f(x2)","b-a"]
				result=IE.table(GoldenSection_page,head,60)
				for i in range(0,n):
					liste=res[i]
					result.insert(parent='',index=i,iid=i,text='',values=liste)
					i+=1
				result.place(x=115,y=450)
				IE.plot(user_input,m-2,k+2,GoldenSection_page,str(deriv1),xtab)

		except Exception as e:
			print(e)
			mbox.showerror('Exception!','Missing or invalid inputs \nNB : a, b, & Tolerance must be a numbers\n      Tolerance must be positif')

#################################################################################################################
# la page de Newton Unidimentionnel en utilisant les outils de tkinter
def Newtonn():
	global Entry_a,c,Input_a,Input_N,Entry_N,user_input,Entry_f,Input_f,Newton_page,root
	Newton_page=IE.create_page("Newto Algorithm","#2f4f4f")
	Newton_page.tkraise()
	Title=Label(Newton_page,text="Newton Method",font="Helvectica 13",bg='#2f4f4f',fg='white')
	Title.place(x=650, y=15)
	Input_f = Label(Newton_page, text="Enter f(x):",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_f.place(x=450, y=50) 
	Entry_f= ttk.Entry(Newton_page , font="Helvectica 12",width=30)
	Entry_f.place(x=600, y=50)
	Input_a = Label(Newton_page, text="Enter x0:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_a.place(x=450, y=90) 
	Entry_a= ttk.Entry(Newton_page , font="Helvectica 12",width=30)
	Entry_a.place(x=600, y=90)
	Input_N = Label(Newton_page, text="Enter b (b>a):",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_N.place(x=450, y=130) 
	Entry_N = ttk.Entry(Newton_page , font="Helvectica 12",width=30)
	Entry_N.place(x=600, y=130)
	btn1 = Button(Newton_page, text = 'Back',bg="#5f9ea0",fg="white",font="Helvectica 9",width="9",command=root.tkraise)
	btn1.place(x=600, y=170)
	btn2 = Button(Newton_page, text = 'Apply',bg="#5f9ea0",font="Helvectica 9",fg="white",width="9",command=showResultNewton)
	btn2.place(x=805, y=170)
	graph=Label(Newton_page,text="Graph of f(x) & f'(x)",bg="white",fg='#2f4f4f',font="Courier 11")
	graph.place(x=915,y=330)
	iterations=Label(Newton_page,text="Iterations table",bg="white",fg='#2f4f4f',font="Courier 11")
	iterations.place(x=280,y=330)
	Newton_page.mainloop()
# la fonction qui nous aide pour appeler l'algorithme de Newton et storer son resulat pour l'afficher en forme d'iteration et graphe
def showResultNewton():
	global Entry_a,Entry_b,c,Input_a,Input_b,user_input,Entry_f,Input_f,Newton_page
	user_input=Entry_f.get()
	if (len(Entry_f.get() )==0):
		mbox.showinfo('Error!','Function entry is empty! Pleas input a function!')
	else:
		try:
			x0=float(Entry_a.get())
			N=int(Entry_N.get())
			animation=[]
			animation.append(x0)
			Newt=NW.NewtonMethod(user_input,x0,N)
			m=x0
			x=smp.Symbol('x')
			deriv1= smp.diff(user_input,x)
			res=Newt.Newton()
			if res==0:
				mbox.showerror('Exception',"Invalid interval !\nNB: a must be bellow to b")
			else:
				for anim in res:
					animation.append(anim[1])
					xtab= np.array(animation)
				n=Newt.N
				#print(n)
				head=['k','xk',"f(xk)","f'(xk)","f''(xk)","dk"]
				result=IE.table(Newton_page,head,60)
				for i in range(0,n):
					liste=res[i]
					result.insert(parent='',index=i,iid=i,text='',values=liste)
					i+=1
				result.place(x=115,y=450)
				IE.plot(user_input,m-10,m+10,Newton_page,str(deriv1),xtab)

		except Exception as e:
			print(e)
			mbox.showerror('Exception!','Missing or invalid inputs \nNB : a, b, & Tolerance must be a numbers\n      Tolerance must be positif')
##################################################################################################################
#################################################################################################################
# la page de Quasi Newton DFP en utilisant les outils de tkinter
def QuasiNewton_DFP():
	global Entry_a,Input_a,Entry_tol,Input_tol,user_input,Entry_f,Input_f,DFP_page
	DFP_page=IE.create_page("Quasi Newton DFP Algorithm","#2f4f4f")
	DFP_page.tkraise()
	Title=Label(DFP_page,text="Davidon–Fletcher–Powell Method",font="Helvectica 13",bg='#2f4f4f',fg='white')

	Title.place(x=650, y=15)
	Input_f = Label(DFP_page, text="Enter f(x):",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_f.place(x=450, y=50) 
	Entry_f= ttk.Entry(DFP_page , font="Helvectica 12",width=30)
	Entry_f.place(x=600, y=50)
	Input_a = Label(DFP_page, text="Enter x0:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_a.place(x=450, y=90) 
	Entry_a= ttk.Entry(DFP_page , font="Helvectica 12",width=30)
	Entry_a.place(x=600, y=90)
	Input_tol = Label(DFP_page, text="Enter Tolerance:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_tol.place(x=450, y=170) 
	Entry_tol = ttk.Entry(DFP_page,font="Helvectica 12",width=30)
	Entry_tol.place(x=600, y=170)
	btn1 = Button(DFP_page, text = 'Back',bg="#5f9ea0",fg="white",font="Helvectica 9",width="9",command=Back)
	btn1.place(x=600, y=210)
	btn2 = Button(DFP_page, text = 'Apply',bg="#5f9ea0",font="Helvectica 9",fg="white",width="9",command=showResultDFP)
	btn2.place(x=805, y=210)
	graph=Label(DFP_page,text="Graph of f(x) & f'(x)",bg="white",fg='#2f4f4f',font="Courier 11")
	graph.place(x=915,y=370)
	iterations=Label(DFP_page,text="Iterations table",bg="white",fg='#2f4f4f',font="Courier 11")
	iterations.place(x=280,y=370)
	DFP_page.mainloop()


# la fonction qui nous aide pour appeler l'algorithme de quasi newton dfp et storer son resulat pour l'afficher en forme d'iteration et 3d
def showResultDFP():
    global Entry_a,Entry_b,Input_a,Input_b,Entry_tol,Input_tol,user_input,Entry_f,Input_f,DFP_page
    user_input=Entry_f.get()
    if (len(Entry_f.get() )==0):
        mbox.showinfo('Error!','Function entry is empty! Please input a function!')
    else:
        a=Entry_a.get()
        tol=float(Entry_tol.get())
        quasi_dfp=QN.DFP(user_input,a,tol)
        res=quasi_dfp.QuasiNewton_DFP()
        n=quasi_dfp.i
        head=['i','x_k',"f(x_k)","Le Pas"]
        result=IE.table(DFP_page,head,140)
        for i in range(0,n):
            liste=res[i]
            result.insert(parent='',index=i,iid=i,text='',values=liste)
        result.place(x=115,y=450)
        quasi_dfp.QuasiNewton_DFP3d()
        


#################################################################################################################
#################################################################################################################
# la page de Quasi Newton BFGS en utilisant les outils de tkinter
def QuasiNewton_BFGS():
	global Entry_a,Input_a,Entry_tol,Input_tol,user_input,Entry_f,Input_f,BFGS_page
	BFGS_page=IE.create_page("Quasi Newton BFGS Algorithm","#2f4f4f")
	BFGS_page.tkraise()
	Title=Label(BFGS_page,text="Broyden–Fletcher–Goldfarb–Shanno Method",font="Helvectica 13",bg='#2f4f4f',fg='white')

	Title.place(x=650, y=15)
	Input_f = Label(BFGS_page, text="Enter f(x):",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_f.place(x=450, y=50) 
	Entry_f= ttk.Entry(BFGS_page , font="Helvectica 12",width=30)
	Entry_f.place(x=600, y=50)
	Input_a = Label(BFGS_page, text="Enter x0:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_a.place(x=450, y=90) 
	Entry_a= ttk.Entry(BFGS_page , font="Helvectica 12",width=30)
	Entry_a.place(x=600, y=90)
	Input_tol = Label(BFGS_page, text="Enter Tolerance:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_tol.place(x=450, y=170) 
	Entry_tol = ttk.Entry(BFGS_page,font="Helvectica 12",width=30)
	Entry_tol.place(x=600, y=170)
	btn1 = Button(BFGS_page, text = 'Back',bg="#5f9ea0",fg="white",font="Helvectica 9",width="9",command=Back)
	btn1.place(x=600, y=210)
	btn2 = Button(BFGS_page, text = 'Apply',bg="#5f9ea0",font="Helvectica 9",fg="white",width="9",command=showResultBFGS)
	btn2.place(x=805, y=210)
	graph=Label(BFGS_page,text="Graph of f(x) & f'(x)",bg="white",fg='#2f4f4f',font="Courier 11")
	graph.place(x=915,y=370)
	iterations=Label(BFGS_page,text="Iterations table",bg="white",fg='#2f4f4f',font="Courier 11")
	iterations.place(x=280,y=370)
	BFGS_page.mainloop()


# la fonction qui nous aide pour appeler l'algorithme de BFGS et storer son resulat pour l'afficher en forme d'iteration et 3d
def showResultBFGS():
    global Entry_a,Entry_b,Input_a,Input_b,Entry_tol,Input_tol,user_input,Entry_f,Input_f,BFGS_page
    user_input=Entry_f.get()
    if (len(Entry_f.get() )==0):
        mbox.showinfo('Error!','Function entry is empty! Please input a function!')
    else:
        a=Entry_a.get()
        tol=float(Entry_tol.get())
        quasi_bfgs=QN.BFGS(user_input,a,tol)
        res=quasi_bfgs.BFGS_method()
        n=quasi_bfgs.i
        
        head=['i','x_k',"f(x_k)","Le Pas"]
        result=IE.table(BFGS_page,head,140)
        for i in range(0,n):
            liste=res[i]
            result.insert(parent='',index=i,iid=i,text='',values=liste)
        result.place(x=115,y=450)
        quasi_bfgs.BFGS3d()
        


#################################################################################################################
#################################################################################################################
# la page de Quasi Newton BFGS par le pas de Goldenstein en utilisant les outils de tkinter
def QuasiNewton_BFGS1():
	global Entry_a,Input_a,Entry_tol,Input_tol,user_input,Entry_f,Input_f,BFGS_page
	BFGS_page=IE.create_page("Quasi Newton BFGS Algorithm Avec Pas GoldStein","#2f4f4f")
	BFGS_page.tkraise()
	Title=Label(BFGS_page,text="Broyden–Fletcher–Goldfarb–Shanno Method Avec Pas GoldStein",font="Helvectica 13",bg='#2f4f4f',fg='white')

	Title.place(x=650, y=15)
	Input_f = Label(BFGS_page, text="Enter f(x):",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_f.place(x=450, y=50) 
	Entry_f= ttk.Entry(BFGS_page , font="Helvectica 12",width=30)
	Entry_f.place(x=600, y=50)
	Input_a = Label(BFGS_page, text="Enter x0:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_a.place(x=450, y=90) 
	Entry_a= ttk.Entry(BFGS_page , font="Helvectica 12",width=30)
	Entry_a.place(x=600, y=90)
	Input_tol = Label(BFGS_page, text="Enter Tolerance:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_tol.place(x=450, y=170) 
	Entry_tol = ttk.Entry(BFGS_page,font="Helvectica 12",width=30)
	Entry_tol.place(x=600, y=170)
	btn1 = Button(BFGS_page, text = 'Back',bg="#5f9ea0",fg="white",font="Helvectica 9",width="9",command=Back)
	btn1.place(x=600, y=210)
	btn2 = Button(BFGS_page, text = 'Apply',bg="#5f9ea0",font="Helvectica 9",fg="white",width="9",command=showResultBFGS1)
	btn2.place(x=805, y=210)
	graph=Label(BFGS_page,text="Graph of f(x) & f'(x)",bg="white",fg='#2f4f4f',font="Courier 11")
	graph.place(x=915,y=370)
	iterations=Label(BFGS_page,text="Iterations table",bg="white",fg='#2f4f4f',font="Courier 11")
	iterations.place(x=280,y=370)
	BFGS_page.mainloop()


# la fonction qui nous aide pour appeler l'algorithme de BFGS par goldstein et stocker son resulat pour l'afficher en forme d'iteration et 3d
def showResultBFGS1():
    global Entry_a,Entry_b,Input_a,Input_b,Entry_tol,Input_tol,user_input,Entry_f,Input_f,BFGS_page
    user_input=Entry_f.get()
    if (len(Entry_f.get() )==0):
        mbox.showinfo('Error!','Function entry is empty! Please input a function!')
    else:
        a=Entry_a.get()
        tol=float(Entry_tol.get())
        quasi_bfgs=QN.BFGS(user_input,a,tol)
        res=quasi_bfgs.BFGS_method()
        n=quasi_bfgs.i
        
        head=['i','x_k',"f(x_k)","Le Pas"]
        result=IE.table(BFGS_page,head,140)
        for i in range(0,n):
            liste=res[i]
            result.insert(parent='',index=i,iid=i,text='',values=liste)
        result.place(x=115,y=450)
        quasi_bfgs.BFGS3d()
        
#################################################################################################################
#################################################################################################################
# la page de Quasi Newton DFP en utilisant les outils de tkinter
def QuasiNewton_SR1():
	global Entry_a,Input_a,Entry_tol,Input_tol,user_input,Entry_f,Input_f,SR1_page
	SR1_page=IE.create_page("Quasi Newton SYMMETRIC-RANK-1 Algorithm","#2f4f4f")
	SR1_page.tkraise()
	Title=Label(SR1_page,text="SYMMETRIC-RANK-1 Method",font="Helvectica 13",bg='#2f4f4f',fg='white')

	Title.place(x=650, y=15)
	Input_f = Label(SR1_page, text="Enter f(x):",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_f.place(x=450, y=50) 
	Entry_f= ttk.Entry(SR1_page , font="Helvectica 12",width=30)
	Entry_f.place(x=600, y=50)
	Input_a = Label(SR1_page, text="Enter x0:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_a.place(x=450, y=90) 
	Entry_a= ttk.Entry(SR1_page , font="Helvectica 12",width=30)
	Entry_a.place(x=600, y=90)
	Input_tol = Label(SR1_page, text="Enter Tolerance:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_tol.place(x=450, y=170) 
	Entry_tol = ttk.Entry(SR1_page,font="Helvectica 12",width=30)
	Entry_tol.place(x=600, y=170)
	btn1 = Button(SR1_page, text = 'Back',bg="#5f9ea0",fg="white",font="Helvectica 9",width="9",command=Back)
	btn1.place(x=600, y=210)
	btn2 = Button(SR1_page, text = 'Apply',bg="#5f9ea0",font="Helvectica 9",fg="white",width="9",command=showResultSR1)
	btn2.place(x=805, y=210)
	graph=Label(SR1_page,text="Graph of f(x) & f'(x)",bg="white",fg='#2f4f4f',font="Courier 11")
	graph.place(x=915,y=370)
	iterations=Label(SR1_page,text="Iterations table",bg="white",fg='#2f4f4f',font="Courier 11")
	iterations.place(x=280,y=370)
	SR1_page.mainloop()


# la fonction qui nous aide pour appeler l'algorithme de quasi newton SR1 et stocker son resulat pour l'afficher en forme d'iteration et 3d
def showResultSR1():
    global Entry_a,Entry_b,Input_a,Input_b,Entry_tol,Input_tol,user_input,Entry_f,Input_f,SR1_page
    user_input=Entry_f.get()
    if (len(Entry_f.get() )==0):
        mbox.showinfo('Error!','Function entry is empty! Please input a function!')
    else:
        a=Entry_a.get()
        tol=float(Entry_tol.get())
        quasi_sr1=QN.SR1(user_input,a,tol)
        res=quasi_sr1.SR1()
        n=quasi_sr1.i
        head=['i','x_k',"f(x_k)","Le Pas"]
        result=IE.table(SR1_page,head,140)
        for i in range(0,n):
            liste=res[i]
            result.insert(parent='',index=i,iid=i,text='',values=liste)
        result.place(x=115,y=450)
        quasi_sr1.SR1_3d()
        


#################################################################################################################
#La méthode qui crée la page de Newton concernant les formulaire où l'utilisateur 
#peut saisir les entrées de l'algorithme et les résultats seront affichées
def Newton2():
	global Entry_a,c,Input_a,Entry_tol,Input_tol,user_input,Entry_f,Input_f,Entry_n,Input_n,Newton2_page,root
	Newton2_page=IE.create_page("Newton Algorithm","#2f4f4f")
	Newton2_page.tkraise()
	Title=Label(Newton2_page,text="Newton Method",font="Helvectica 13",bg='#2f4f4f',fg='white')
	Title.place(x=650, y=15)
	Input_n = Label(Newton2_page, text="Enter number of variables:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_n.place(x=420, y=50) 
	# l'epace où l'utilisateur peut saisir le nombre de variables
	Entry_n= ttk.Entry(Newton2_page , font="Helvectica 12",width=30)
	Entry_n.place(x=600, y=50)
	Input_f = Label(Newton2_page, text="Enter f(x1,x2,..):",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_f.place(x=450, y=90) 
	# l'epace où l'utilisateur peut saisir la fonction
	Entry_f= ttk.Entry(Newton2_page , font="Helvectica 12",width=30)
	Entry_f.place(x=600, y=90)
	# l'epace où l'utilisateur peut saisir le point initial
	Input_a = Label(Newton2_page, text="Enter x0:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_a.place(x=450, y=130) 
	Entry_a= ttk.Entry(Newton2_page , font="Helvectica 12",width=30)
	Entry_a.place(x=600, y=130)
	Input_tol = Label(Newton2_page, text="Enter Tolerance:",font="Helvectica 12",bg='#2f4f4f',fg='white')
	Input_tol.place(x=450, y=170) 
	# l'epace où l'utilisateur peut saisir la tolérance
	Entry_tol = ttk.Entry(Newton2_page,font="Helvectica 12",width=30)
	Entry_tol.place(x=600, y=170)
	# les buttons pour executer l'algorithme ou revenir à la page précédante
	btn1 = Button(Newton2_page, text = 'Back',bg="#5f9ea0",fg="white",font="Helvectica 9",width="9",command=root.tkraise)
	btn1.place(x=600, y=210)
	btn2 = Button(Newton2_page, text = 'Apply',bg="#5f9ea0",font="Helvectica 9",fg="white",width="9",command=showResultNewton2)
	btn2.place(x=805, y=210)
	graph=Label(Newton2_page,text="Graph of f(x) & f'(x)",bg="white",fg='#2f4f4f',font="Courier 11")
	graph.place(x=915,y=370)
	iterations=Label(Newton2_page,text="Iterations table",bg="white",fg='#2f4f4f',font="Courier 11")
	iterations.place(x=280,y=370)
	Newton2_page.mainloop()


# La fonction qui permet de récuperer les entrée saisit par l'utilisateur,
# executer l'algorithme de newton et afficher le résultat 
def showResultNewton2():
    global Entry_a,Input_a,Entry_tol,Input_tol,Entry_n,Input_n,user_input,Entry_f,Input_f,Newton2_page
    user_input=Entry_f.get()
    #si la fonction n'est pas saisit, une boite d'erreur sera affiché
    if (len(Entry_f.get() )==0):
        mbox.showinfo('Error!','Function entry is empty! Please enter a function!')
    else:
        a=Entry_a.get()# récuperer le point initiale
        tol=float(Entry_tol.get())# récupérer le tolérance
        nv=int(Entry_n.get()) # récupérer le nombre de variables
        newton2=NW2.Newton2(user_input,a,tol,nv) # créer un objet de la classe de Newton en multidimentionnel
        res=newton2.Newton2() # exécuter la méthode de newton 
        n=newton2.i # le nombre d'itérations
        head=['i','x_k',"f(x_k)","Le Pas"]
        result=IE.table(Newton2_page,head,150) 
        #la table d'itérations
        for i in range(0,n):
            liste=res[i]
            result.insert(parent='',index=i,iid=i,text='',values=liste)
        result.place(x=115,y=450)
        #IE.plot(user_input,m-2,m+10,FaussePosition_page,str(deriv1),xtab)
        # si le nombre de variables est 2 on affiche le graphe en 3D
        if nv==2:
        	newton2.GradientDescent3d()
#################################################################################################################

#################################################################################################################
# La page d'acceuil avec le menu de tous les algorithmes développé dans ce projet 
root =IE.create_page("Home","#2f4f4f")
my_menu=Menu(root)
root.config(menu=my_menu)
Title=Label(root,text="Welcome To The Optimization's World",font="Helvectica 30",bg='#2f4f4f',fg='white')
Title.place(x=390, y=250) #Le titre de l'application et son emplacement
###########################
#Les Menus des options
###########################
menubar = Menu(root,tearoff=0,background='beige', foreground='black',activebackground='#004c99', activeforeground='white') # Menu principale

unid = Menu(menubar) # Menu de Unidimentionnel

multi = Menu(menubar) # Menu de Multidimentionnel

Quit = Menu(menubar) # Menu pour Quitter

pas_app = Menu(multi) # Menu du pas Approche

pas_exact = Menu(multi) # Menu Du pas Exacte

QuasiNewton_exact=Menu(pas_exact)

# chaque option est nomme selon son algorithme approprie ( Gradient, Conjugue ,Newton )
pas_exact.add_command(label='Gradient',font=("Helvetica", 11),command =Grad_Descent )
pas_exact.add_command(label='Conjugate Gradient',font=("Helvetica", 11))
pas_exact.add_command(label='Newton',font=("Helvetica", 11),command = Newton2)
######################################################
pas_exact.add_cascade(label='Quasi Newton',menu=QuasiNewton_exact,font=("Helvetica", 14)) #Cascade du Quasi Newton

# chaque option est nomme selon son algorithme approprie ( DFP, BFGS , SR1 )
QuasiNewton_exact.add_command(label='Davidon–Fletcher–Powell',font=("Helvetica", 14),command=QuasiNewton_DFP)
QuasiNewton_exact.add_command(label='Broyden–Fletcher–Goldfarb–Shanno',font=("Helvetica", 14),command=QuasiNewton_BFGS)
QuasiNewton_exact.add_command(label='SR1',font=("Helvetica", 14),command=QuasiNewton_SR1)
########################################################################################

quitter=Menu(Quit) #Boutton pour Quitter

#Les menus de Pas Approchee
gradient=Menu(pas_app)
Conjugate=Menu(pas_app)
Newton=Menu(pas_app)
QuasiNewton=Menu(pas_app)
##########################

QuasiNewtonDFP=Menu(QuasiNewton)
QuasiNewtonBFGS=Menu(QuasiNewton)
######################## Les cascades des pas Approchee #########################
pas_app.add_cascade(label='Gradient', menu=gradient,font=("Helvetica", 11)) #GrADIENT
pas_app.add_cascade(label='Conjugate Gradient', menu=Conjugate,font=("Helvetica", 11)) #conjuguee
pas_app.add_cascade(label='Newton', menu=Newton,font=("Helvetica", 11)) #Newton
pas_app.add_cascade(label='Quasi Newton', menu=QuasiNewton,font=("Helvetica", 11)) #Quasi Newton
#################################################################################

###############Les commandes Relie Avec chaque Pas Approchee##############################################
gradient.add_command(label='Armijo',font=("Helvetica", 11),command=GradArmijo) #Armijo Pour le Gradient
gradient.add_command(label='GoldStein',font=("Helvetica", 11),command=GradGold) # GoldenStein Pour le Gradient
gradient.add_command(label='Wolfe',font=("Helvetica", 11),command=Bissection) # Wolfe Pour le Gradient

Conjugate.add_command(label='Armijo',font=("Helvetica", 11),command=Bissection) #Armijo Pour le Conjuguee
Conjugate.add_command(label='GoldStein',font=("Helvetica", 11),command=Bissection)  # GoldenStein Pour le Conjuguee
Conjugate.add_command(label='Wolfe',font=("Helvetica", 11),command=Bissection) # Wolfe Pour le Gradient le Conjuguee

Newton.add_command(label='Armijo',font=("Helvetica", 11),command=Bissection) #Armijo Pour Newton
Newton.add_command(label='GoldStein',font=("Helvetica", 11),command=Bissection) # GoldenStein Pour Newton
Newton.add_command(label='Wolfe',font=("Helvetica", 11),command=Bissection) # Wolfe Pour Newton

QuasiNewtonDFP.add_command(label='Armijo',font=("Helvetica", 11),command=Bissection)  #Armijo Pour DFP
QuasiNewtonDFP.add_command(label='GoldStein',font=("Helvetica", 11),command=Bissection) # GoldenStein Pour DFP
QuasiNewtonDFP.add_command(label='Wolfe',font=("Helvetica", 11),command=Bissection) # Wolfe Pour DFD


QuasiNewtonBFGS.add_command(label='Armijo',font=("Helvetica", 11),command=Bissection) #Armijo Pour BFGS
QuasiNewtonBFGS.add_command(label='GoldStein',font=("Helvetica", 11),command=QuasiNewton_BFGS1)  # GoldenStein Pour BFGS
QuasiNewtonBFGS.add_command(label='Wolfe',font=("Helvetica", 11),command=Bissection) # Wolfe Pour BFGS
##########################################################################################################


############################# Cascades Pour QuasiNewton ( DFP et BFGS )####################################
QuasiNewton.add_cascade(label='Davidon–Fletcher–Powell',menu=QuasiNewtonDFP,font=("Helvetica", 11))
QuasiNewton.add_cascade(label='Broyden–Fletcher–Goldfarb–Shanno',menu=QuasiNewtonBFGS,font=("Helvetica", 11))
###########################################################################################################

multi.add_cascade(label='Pas Approchee', menu=pas_app,font=("Helvetica", 11))
multi.add_cascade(label='Pas Exacte', menu=pas_exact,font=("Helvetica", 11))


unid.add_command(label='Bissection',font=("Helvetica", 11),command=Bissection)
unid.add_command(label='Fibonacci',font=("Helvetica", 11),command=Fibonacci)
unid.add_command(label='Golden Section',font=("Helvetica", 11),command=GoldenSection)
unid.add_command(label='Newton',font=("Helvetica", 11),command=Newtonn)
unid.add_command(label='Fausse Position',font=("Helvetica", 11),command=FaussePosition)

# si cliquer on quite l'interface
Quit.add_command(label="Quit",font=("Helvetica", 11), command=root.quit)


menubar.add_cascade(label="Unidimentionnel", menu=unid)
menubar.add_cascade(label="Multidimentionnel", menu=multi)
menubar.add_cascade(label="Quit", menu=Quit)
menubar.config( font=("Helvetica", 11))
###########################
#Les Menus des options
###########################

#menubar2.config( font=("Helvetica", 14))
root.config(menu=menubar)
#root.config(menu=menubar2)



root.mainloop()
