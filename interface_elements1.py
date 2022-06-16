import sympy as smp
import numpy as np
from math import *
import tkinter as tk
from  tkinter import ttk
from tkinter import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.animation import FuncAnimation


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

#Pour creer une nouvelle page au niveau de l'interface
def create_page(title,bg_color):
    root = Tk()
    root.state('normal')
    root.rowconfigure(0,weight=1)
    root.columnconfigure(0,weight=1)
    root.title(title)
    root.configure(bg=bg_color)
    return root

def show_frame(frame):
	frame.tkraise()


def button(text , clmn , rw):
	ttk.Button(root , text=text , command = lambda:insert(text)).grid(column = clmn , row =rw , ipady = 6 , ipadx=1,font="relway",bg="cyan",bd="white")

#Pour afficher le tableau des iterations
def table(frame,head,wid):
    my_table = ttk.Treeview(frame)
    my_table['columns']=head
    my_table.column("#0", width=2,  stretch=YES)
    for i in head:
        my_table.column(i,anchor=CENTER, width=wid)
        my_table.heading(i,text=i,anchor=CENTER)

    return my_table
# Pour ploter les algorithmes de unidimentionnel et leurs propres iterations
def plot(*args):
    def xfunction(x,input):
        return eval(input)
    def fonction(input):
        def fonc(x):
            return eval(input)
        return fonc
    xAxis= np.linspace(args[1],args[2],100)
    val=abs(args[1]-args[2])
    fig= Figure(figsize=(6,4),dpi=60)
    a=fig.add_subplot(1,1,1)
    a.spines['left'].set_position('zero')
    a.spines['right'].set_color('none')
    a.spines['bottom'].set_position('zero')
    a.spines['top'].set_color('none')


    f = np.vectorize(xfunction)
    a.plot(xAxis,f(xAxis,args[0]),label="f(x)")
    a.plot(xAxis,f(xAxis,args[4]),label="f'(x)")
    a.legend()
    canvas=FigureCanvasTkAgg(fig,args[3])
    canvas.draw()
    canvas.get_tk_widget().place(x=700,y=100)
    canvas._tkcanvas.place(x=840,y=450)
	    
    f=fonction(args[0])

    xlist=np.arange(-10,10.1,0.01)
    # l'axe des ordonnees 
    ylist=f(xlist)
    # declarer une figure ou on va afficher le graphe
    fig=plt.figure(num=0,dpi=120,figsize=(14, 8))
    # l'affichage du graphe seulement
    plt.plot(xlist,ylist)
    # On va afficher un titre qui represente le grafhe avec le nom d'algorithme en question   
    #plt.title("Application de la Fausse Position  Pour trouver le minimum", fontsize=14)
    plt.xlabel('X', fontsize=13)
    plt.xticks(fontsize=9)
    plt.ylabel('F(X)', fontsize=13)
    plt.yticks(fontsize=9)

    
    graph, = plt.plot([], [], 'o')
    xtab=args[5]
    def animate(i):
        Image=f(xtab)
        graph.set_data(xtab[:i+1], Image[:i+1],)
        return graph,

    ani = FuncAnimation(fig, animate, frames=15, interval=500, blit=True, repeat=False)
    plt.show()

