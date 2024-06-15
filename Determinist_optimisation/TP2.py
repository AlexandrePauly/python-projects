#### on commence par installer les librairies nécessaires
import numpy as np                     # permet d'effectuer des calculs numériques avec Python
import matplotlib.pyplot as plt        # permet de tracer des graphes et de les manipuler. Elle effectue un certain nombre de traitements pour préparer l'affichage du graphique.
import random                          # Permet la génération de nombres aléatoires.

#########################  Fonction quadratique  ########################

# Fonction quadratique
def f_quadra(x,A,b,c):
  return c + np.dot(np.transpose(b),x) + 0.5*np.dot(np.dot(np.transpose(x),A), x)

# Fonction quadratique (avec les variables globales A, b, c et x)
def fq(x):
  return c + np.dot(np.transpose(b),x) + 0.5*np.dot(np.dot(np.transpose(x),A), x)

# Gradient de la fonction quadratique
def grad_f_quadra(x,A,b):
  return np.dot(A,x) + b

# Gradient de la fonction quadratique (avec les variables globales A, b et x)
def grad_fq(x):
  return np.dot(A,x) + b

######################## Fonction des moindres carrés ########################

# Fonction des moindres carrés 
def f_moindresC(x,B,beta):
  return 0.5*np.linalg.norm(beta - np.dot(B,x))*np.linalg.norm(beta - np.dot(B,x))

# Fonction des moindres carrés dratique (avec les variables globales B, beta et x)
def fm(x):
  return 0.5*np.linalg.norm(beta - np.dot(B,x))*np.linalg.norm(beta - np.dot(B,x))

# Gradient de la fonction des moindres carrés 
def grad_f_moindresC(x,B,beta):
  return np.dot(np.transpose(B),(np.dot(B,x) - beta))

# Gradient de la fonction des moindres carrés (avec les variables globales B, beta et x)
def grad_fm(x):
  return np.dot(np.transpose(B),(np.dot(B,x) - beta))

#############  Les données pour la fonction quadratique et la fonction des moindres carrés ######################

#Initialisation de variables
A = np.array([[10,0],[0,1]])      # Matrice symétrique d'ordre n
b = np.array([-3,-3])             # Vecteur de R^n
c = 0                             # Réel
B = np.array([[1,0],[0,1],[1,1]]) # Matrice d'ordre m x n
beta = np.array([6,3,3])          # Vecteur de R^m
x0 = np.array([-2,-7])            # Vecteur de R^n

## Exercice 2 -- Compiler ce code pour vérifier que vos fonctions de l'exercice 1 renvoient les bonnes valeurs

##1ère question
f0 = f_quadra(x0,A,b,c)
f00 = fq(x0)
gradf0 = grad_f_quadra(x0,A,b)
gradf00 = grad_fq(x0)

##2ème question
g0 = f_moindresC(x0,B,beta)
g00 = fm(x0)
gradg0 = grad_f_moindresC(x0,B,beta)
gradg00 = grad_fm(x0)

# Affichage des résultats
print("f0 = f(x0) =", f0,
      ",\tf00 = f(x0) =", f00,
      ",\ngradf0 = ∇f(x0) =", gradf0,
       ",\tgradf00 = ∇f(x0) =", gradf00,
           ".")
print("g0 = g(x0) =" , g0 ,
      ",\tg00 = g(x0) =" , g00 ,
      ",\ngradg0 = ∇g(x0) =" , gradg0,
       ",\tgradg00 = ∇g(x0) =" , gradg00,
           ".")

################ Descente de gradient à pas fixe
def descente_G_fixe(x0,f,grad_f,pas,PRECISION,ITER_MAX):
  # Initialisation des variables
  k = 0   # Indice de boucle
  xk = x0 # Point de départ
  
  # Tant que le nombre d'itération max n'est pas atteint ou que le norme du gradient de la fonction est plus petit que notre précision,
  while k < ITER_MAX and np.linalg.norm(grad_f(xk)) > PRECISION:
    dk = -grad_f(xk) # Direction de descente
    xk = xk + pas*dk # Mise à jour de xk avec sa position actuelle
    k = k + 1        # Incrémentation de l'indice

  # Affichage de messages
  print("\nLe nombre d'itérations est k =", k)
  print("La norme du gradient en l'itéré xk, ∥∇f(xk)∥ =", np.linalg.norm(grad_f(xk)))

  return print("La solution obtenue est xk =", xk)

## Données critère arrêt
PRECISION = 10**-5
ITER_MAX = 200

# Exécuter ce code pour répondre à l'exercice 4
pas = 0.12
descente_G_fixe(x0,fq,grad_fq,pas,PRECISION,ITER_MAX)

################ Descente de gradient à pas optimal pour une fonction quadratique
def descente_G_Cauchy(x0,fq,grad_fq,PRECISION,ITER_MAX):
  # Initialisation des variables
  k = 0   # Indice de boucle
  xk = x0 # Point de départ

  # Tant que le nombre d'itération max n'est pas atteint ou que le norme du gradient de la fonction est plus petit que notre précision,
  while k < ITER_MAX and np.linalg.norm(grad_fq(xk)) > PRECISION:
    k = k + 1                                                                              # Incrémentation de l'indice
    dk = -grad_fq(xk)                                                                      # Direction de descente
    rhok = -np.dot(np.transpose(dk),grad_fq(xk)) / np.dot(np.dot(np.transpose(dk),A),dk)   # 
    xk = xk + rhok*dk                                                                      # Mise à jour de xk avec sa position actuelle
    erreur = np.linalg.norm(grad_fq(xk))                                                   # Critère d'arrêt

  # Affichage de messages
  print("\nPour la forme quadratique f de matrice A = ",A ,",b = ",b ,"et c =", c)
  print("En initialisant avec x0 =", x0)
  print("On obtient les résultats suivants avec un pas optimal de Cauchy:", rhok)
  print("Le critère d'arrêt est satisfait en ",k ,"itérations avec un gradient de norme égale à", np.round(erreur,5))
  print("La solution optimale obtenue est :", xk)

  return

# Appel de la fonction de descente par la méthode de Cauchy
descente_G_Cauchy(x0,fq,grad_fq,PRECISION,ITER_MAX)

### Pas d'Armijo à l'itération k
def Armijo(f,grad_f,x,d):
  # Initialisation des variables
  tau = random.random() # On le choisit de façon aléatoire
  w = 10**(-4)          # Précision
  rho = 0.7             # Pas initial

  while f(x + rho * d) > f(x) + w * rho * np.dot(grad_f(x), d):
    rho = random.uniform(tau * rho, (1 - tau) * rho)  # Choix du pas dans l'intervalle [tau*rho, (1-tau)*rho]
  
  return rho

### Gradient par la méthode d'Armijo
def gradient_Armijo(x0,f,grad_f,PRECISION,ITER_MAX):
  # Initialisation des variables
  xk = x0          # Point de départ
  pas_history = [] # 
  k = 0            # Indice de boucle

  # Tant que le nombre d'itération max n'est pas atteint ou que le norme du gradient de la fonction est plus petit que notre précision,
  while k < ITER_MAX and np.linalg.norm(grad_f(xk)) > PRECISION:
    k = k +1                            # Incrémentation de l'indice
    dk = -grad_f(xk)                    # Direction de descente
    rhok = Armijo(f, grad_f,x,dk)       # 
    xk = xk + rhok*dk                   # Mise à jour de xk avec sa position actuelle
    erreur = np.linalg.norm(grad_f(xk)) # Critère d'arrêt
    pas_history.append(rhok)

  # Affichage de messages
  print("La liste des pas choisis à chaque itération est :", pas_history)
  print("Le nombre d'itérations est k =", k)
  print("La solution optimale obtenue est :", xk)
  
  return

Liste = np.array([0.001,0.01,0.05,0.07,0.11])

def graphe1_gradient_fixe(x0,f,grad_f,Liste):
  # Initialisation de variables
  N = len(Liste)                     # Longueur de la liste
  f_history = np.zeros((N,ITER_MAX)) #

  for i in range(N):
    xk = x0
    for k in range(ITER_MAX):
      dk = -grad_f(xk)
      xk = xk + Liste[i]*dk
      f_history[i,k] = f(xk)

  plt.figure(figsize=(15,4))

  for i in range(N):
    plt.plot(f_history[i,:], label = "pas égal à "+str(Liste[i]))

  # Affichage d'un graphique
  plt.title("Descente du gradient à pas fixe -- Convergence vers le minimum de la fonction $f$")
  plt.xlabel("Nombre d'itérations k")
  plt.ylabel("$f(x_k)$")
  plt.grid()
  plt.legend()
  plt.show()

  return

def graphe2_gradient_fixe(x0,f,grad_f,Liste):
  # Initialisation de variables 
  N = len(Liste)                     # Longueur de la liste
  f_history = np.zeros((N,ITER_MAX)) #

  for i in range(N):
    xk = x0
    for k in range(ITER_MAX):
      dk = -grad_f(xk)
      xk = xk + Liste[i]*dk
      f_history[i,k] = np.linalg.norm(grad_f(xk))

  plt.figure(figsize=(15,4))

  for i in range(N):
    plt.plot(f_history[i,:], label = "pas égal à "+str(Liste[i]))

  # Affichage d'un graphique
  plt.title("Descente du gradient à pas fixe -- Convergence vers le minimum de la fonction $f$")
  plt.xlabel("Nombre d'itérations k")
  plt.ylabel("$f(x_k)$")
  plt.grid()
  plt.legend()
  plt.show()

  return

# Appel des fonctions pour afficher les graphes
graphe1_gradient_fixe(x0, fq, grad_fq, Liste)
graphe2_gradient_fixe(x0, fq, grad_fq, Liste)