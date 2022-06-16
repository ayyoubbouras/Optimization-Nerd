# OPTIMIZATION APPLICATION

## Attention
To view the readme.md file clearly visit this website [Here](https://pandao.github.io/editor.md/en.html)

## Description

In our App we are utilizing root solving techniques such as:
### Unidimentionel
- Bisection Method
- Newton's Method
- False Position
- Golden Section
- Fibonacci
- Hybrid of Newton and False Position
- Hybrid of Newton and Bisection

### Multidimentionel

We used both the exact and the inexact steps (Armijo , Goldstein) for the folowing algorithms:
- Gradient Descent
- Conjugate Gradient
- Newtons Method
- Quasi Newton (DFP , BFGS , SR1)


### Instructions

 To Run our program :
- First visit the the folder of the existing program and open your terminal 
- Now run the main.py file in your terminal  -> python main.py
- It's pretty clear from here  , you get a user friendly interface 

###Remarks
1 - we used scipy.misc.derivative(func, x0, dx=1.0, n=1, args=(), order=1) in our single variable diffientials so that we can choose precisly the spacing dx (h) .
2 - the file newton.py contains unidementional newton and newton2 contains multidimentional newton's method
3 - we used numpy.linalg.eig to obtain the eigen values of the hessian matrix to apply the perturbation
4 - we used 3D quiver plot to draw the direction of our iteration ( the descent direction )
5 - for most mathematical calculations and needs we used numpy , sympy and numdifftools
6 - we used FuncAnimation function to draw the 2d iterations

## Special Packages

|  No | Package or Library                                                                                                                                | Link To Documentation                                                 |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| 1     | scipy.misc.derivative                                  | [derivative documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.derivative.html)                  |
| 2     | numpy.linalg.eig                                              | [eig documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html)              |
| 3     | 3D quiver plot                                          | [quiver documentation](https://matplotlib.org/stable/gallery/mplot3d/quiver3d.html)                  |
| 4     | FuncAnimation                                          | [FuncAnimation documentation](https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.animation.FuncAnimation.html)                  |

