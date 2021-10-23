##############################################################################
#
#         Le langage de programmation Python - Introduction
#         Jean-Yves Trépanier et Eddy Petro - Aout 2019
#
##############################################################################

#----------------------------#
#   Importation de modules   #
#----------------------------#  

# importation du module spécialisé "numpy" et association de ce dernier
# à la variable locale np. numpy est recommandé pour le calcul scientifique
import numpy as np

# début de la définition de la fonction pause
def pause():
    PauseProgramme = input("Presser <RETURN> pour continuer...")
# fin de la définition de la fonction pause
    
# faire une pause   
pause()

# Exemples d'appels de méthodes contenues dans un module spécialisé numpy
a = np.absolute(-10)
b = np.pi
c = np.cos(b)
d = np.sin(b/2)
e=np.log(10)
print('Appels divers à Numpy')
print('absolute(-10)',a,'pi',b,'cos(pi)',c,'sin(pi/2)',d,'log(1)',e)

pause()

#----------------------------#
#         Affichage          #
#----------------------------# 


# Afficher un caratère à l'écran
print('*')

# Afficher une répétition de caractères 
print('*' * 10)

# Afficher le contenu d'une variable 
e = 10.00
print('Valeur de e = ',e)

pause()

#-------------------------------------------------#
#      Types de données & conversion de types     #
#-------------------------------------------------#

# Déclaration de types 
x = 13                         # integer1
x = 20.                        # float
x = '10.0'                     # string        

# Déterminer le type d'une variable et afficher
x = 10
print(type(x))

# Modifier le type d'une variable de "integer" à "float"  
print(type(float(x)))

# Modifier le type d'une variable de "float" à "integer"  
print(type(int(x)))

# Modifier le type d'une variable de "integer" à "string"  
print(type(str(x)))

# Concatener une série de caractères et afficher
print('j' + "'" + 'ai ' + str(x) + ' pommes')

pause()

#--------------------------------------#
#        Opérateurs arithmétiques      #
#--------------------------------------# 

# Multiplication
c = 1. * 2

# Exposant
d = 3. ** 3.

# Division (entier)
e = 4 / 2

# Division (réel)
f = 4. / 2

# Division entière (calcul le reste de la division)
g = 5 // 2

# Modulo
h = 12 % 5

print(c,d,e,f,g,h)
pause()

#--------------------------------------#
#       Opérateurs d'assignation       #
#--------------------------------------# 

# Opération x = 3
x = 3
print(x)
# Opération x = x + 3
x += 3
print(x)
# Opération x = x - 3
x = x - 3
print(x)
# Opération x = x * 3
x *= 3
print(x)
# Opération x = x / 3
x /= 3
print(x)
pause()

#--------------------------------------#
#               Tableaux               #
#--------------------------------------# 

# tableau 1D contenant 10 éléments (indexés de 0 à 9)
b = range(0,10)
print(b[0],b[9])

# Définition d'un tableau 1D contenant 2 éléments en utilisant le module "numpy"
b = np.array((0, 0))
print(b)

# Définition d'une matrice 2x2 en utilisant le module "numpy"
A = np.ndarray((2, 2))
A[0,0]=11
A[1,0]=21
A[0,1]=12
A[1,1]=22

print(A)

pause()

# Utilisation de la commande "slice" (Permet de découper ou segmenter une
# structure de donnée séquentielle telle qu'une liste ou un tableau 1D)

nums = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# Exemple 1
some_nums = nums[2:7]       # Contient les éléments indexées de 2 à 6
print(some_nums)

# Exemple 2
some_nums = nums[0:4]       # Contient les éléments indexées de 0 à 3
print(some_nums)

# Exemple 3
some_nums = nums[:5]        # Contient les 5 premiers éléments  du tableau
print(some_nums)

# Exemple 4
some_nums = nums[-3:]       # Contient les 3 derniers éléments  du tableau
print(some_nums)

pause()

#--------------------------------------#
#           Boucles et tests           #
#--------------------------------------# 

#
#  note importante: les indentations à l'intérieur des boucles et tests
#  doivent absolument être respectées. L'omission de celles-ci engendrera
#  des erreurs.
#

# Boucle "for" affichant les chiffres de 0 à 5
for x in range(6):
  print('premier for ', x)
  
# Boucle "for" affichant les chiffres de 2 à 5  
for x in range(2, 6):
  print('deuxieme for ', x)

# Boucle "for" affichant les chiffres de 2 à 29 par incrément de 3  
for x in range(2, 30, 3):
  print('troisieme for ', x)

# Boucle "while" affichant les chiffres de 0 à 5  
i = 0
while i < 6:
  print('while ', i)
  i += 1

pause()

# Conditions "if"   "if else"   "if elif else"

# Opérateurs de comparaisons
  
#    ==      égal à 
#    !=      différent de 
#    <       strictement inférieur à 
#    <=      inférieur ou égal à
#    >       strictement supérieur à 
#    >=      supérieur ou égal à

# Exemple 1 - Condition if
a = 10
if a > 5:
   a = a + 1
print(a)
# Exemple 2 - Condition if else
a = 10
if a > 5:
   a = a + 1
else:
   a = a - 1  
print(a)
# Exemple 3 - Condition if elif else
a = 5
if a > 5:
   a = a + 1
elif a == 5:
   a = a + 10 
else:
   a = a - 1  
print(a)
pause()

#--------------------------------------#
#      Définition d'une fonction       #
#--------------------------------------# 

def f(x):
    y = max(x,0.) + 10 
    return y**2

z = f(2)
print(z)
pause()

#-----------------------------------------------------------------------------#
#                                                                             #  
#                          Linear algebra (numpy.linalg)                      #
#                                                                             #  
# Références: https://docs.scipy.org/doc/numpy/reference/routines.linalg.html #
#                                                                             #  
#-----------------------------------------------------------------------------#

# Génération d'une matrice 3x3 contenant des valeurs aléatoires
A = np.random.rand(3,3)
print(A)

# Génération d'un vecteur colonne contenant des valeurs aléatoires
b = np.random.rand(3,1)
print(b)

# Résolution du système Ax = b en utilisant la fonction "solve" de "numpy" 
x = np.linalg.solve(A, b)
print(x)

# Résolution du système Ax = b en utilisant les fonctions "inv" et "dot" de "numpy" 
x = np.dot(np.linalg.inv(A), b)
print(x)

pause()

#-----------------------------------------------------------------------------#
#                                                                             #  
#                   Intégration & ODEs (scipy.integrate)                      #
#                                                                             #  
# Références: # https://docs.scipy.org/doc/scipy/reference/integrate.html     #
#                                                                             #  
#-----------------------------------------------------------------------------#

#--------------------------------------------------------#
#      Intégrale: méthode du trapèze et de Simpson       #
#--------------------------------------------------------# 

from scipy.integrate import simps

def f(x):
    return x**2

x = np.linspace(0,1,11)
y = f(x)

integral = np.trapz(y,x)       # méthode du trapèze
integralS = simps(y,x)         # méthode de Simpson

print("L'intégrale trapz : ", integral)
print("L'intégrale simps : ", integralS)
print("La solution : ",1./3)

pause()

#--------------------------------------------------------#
#                   Régression linéaire                  #
#--------------------------------------------------------# 

import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(12345678)
x = np.random.random(10)
y = np.random.random(10)

# Regression polynomiale
a,b = np.polyfit(x,y,1)

# Regression linéaire avec statistique
slope, intercept, r_value, p_value, std_err= stats.linregress(x, y)
print("r-squared:", r_value**2)
print("p-value:", p_value)
print("std_err:", std_err)

plt.plot(x,y, 'o')
plt.plot(x,slope*x+intercept)
plt.show()

pause()

#--------------------------------------------------------#
#                      Interpolation                     #
#--------------------------------------------------------# 

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, num=11, endpoint=True)
y = np.cos(x)

f = interp1d(x, y,  kind='linear',fill_value='extrapolate')
f2 = interp1d(x, y, kind='cubic',fill_value='extrapolate')

xnew = np.linspace(0, 11, num=41, endpoint=True)

plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.legend(['data','linear','cubic'], loc='best')
plt.show()

pause()

#--------------------------------------------------------#
#                    Graphique simple                    #
#--------------------------------------------------------# 

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 100)
xx = np.linspace(0, 2*np.pi, 10)

alpha = 1
plt.plot(x, np.sin(alpha*x), label="$\sin(\\alpha x)$")
plt.plot(xx, np.sin(alpha*xx), "s", label="$\sin(\\alpha x)$")
plt.plot(x, np.cos(alpha*x), label="$\cos(\\alpha x)$")
plt.legend()
plt.show()

pause()

#--------------------------------------------------------#
#           Graphique en échelle logarithmique           #
#--------------------------------------------------------# 

import numpy as np
import matplotlib.pyplot as plt


dx = np.array([0.1, 0.02, 0.01, 0.005])
error = 0.5*dx**2
a,b = np.polyfit(np.log(dx), np.log(error), 1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dx,error,'ko',label='$\Vert e_{\mathbf{u}}\Vert_{2}$')

ax.plot(dx, np.exp(b)*dx**a, '-k', label='$\Vert e_{\mathbf{u}}\Vert_{2}= \
        %3.2f \Delta x^{%3.2f}$' %(np.exp(b),a))

ax.set_yscale('log')
ax.set_xscale('log')

ax.grid(b=True, which='minor', color='grey', linestyle='--')
ax.grid(b=True, which='major', color='k', linestyle='-')
plt.ylabel('$\Vert e \Vert_2$')
plt.xlabel('$\Delta x$')
plt.legend()
plt.show()

pause()

#---------------------------------------------------------------------------------------------------------------#
#                                                                                                               #
#                                        Visualisation 2D - Contours                                            #
#                                                                                                               #
#     Référence: https://stackoverflow.com/questions/9008370/python-2d-contour-plot-from-3-lists-x-y-and-rho    #
#                                                                                                               #
#---------------------------------------------------------------------------------------------------------------#

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

# Generate data:
x=np.array((1,2,3,4,5,6,7,8,9))
y=2*x
z=np.cos(x*y)

# Set up a regular grid of interpolation points
xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolate
rbf = scipy.interpolate.Rbf(x, y, z, function='linear')
zi = rbf(xi, yi)

plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
           extent=[x.min(), x.max(), y.min(), y.max()])

plt.colorbar()
plt.show()

