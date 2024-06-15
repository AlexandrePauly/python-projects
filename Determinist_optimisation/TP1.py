# Librairies nécessaires
import numpy as np # permet d'effectuer des calculs numériques avec Python

############### Fonctions ###############

# Fonction f(x) = 2*x
def ft(x,rk):
  return 2*x

# Gradient de la fonction f(x) = 2*x
def grad_ft(x,rk):
  return 2

# Fonction g(x) = 1 - x
def g(x):
    return 1 - x

# Gradient de la fonction g(x) = 1 - x
def grad_g(x):
    return -1

# Fonction barrière inverse
def B(x):
    return -1/g(x)

# Gradient de la fonction barrière inverse
def grad_B(x):
    return grad_g(x)/(g(x)**2)

# Fonction de pénalité
def p(x,rk):
    return ft(x,rk) + rk*B(x)

# Gradient de la fonction de pénalité
def grad_p(x,rk):
    return grad_ft(x,rk) + rk*grad_B(x)

# Fonction barrière logarithmique
def Blog(x):
    return -np.log(-g(x))

# Gradient de la fonction barrière logarithmique
def gradB_log(x):
    return -1/(x - 1)

# Fonction de pénalité
def logp(x,rk):
    return ft(x,rk) + rk*Blog(x + 0.001)

# Gradient de la fonction de pénalité
def grad_logp(x,rk):
    return grad_ft(x,rk) + rk*gradB_log(x + 0.001)

# Fonction pour l'exercice 4
def f_exo4(x,y):
    return 2*x**2 + 3*x*y + 2*y**2

# Gradient de la fonction pour l'exercice 4
def gradf_exo4(x,y):
    return [4*x + 3*y, 3*x + 4*y]

############### Méthodes ###############
    
# Pas d'Armijo à l'itération k avec une constante r
def Armijo(f,grad_f,x,d,rk):
    # Initialisation des variables
    rho = 1
    w = 10**(-4)
    tau = 0.7

    while f(x + rho*d,rk) > f(x,rk) + w * rho * grad_f(x,rk)*d:
        rho = tau * rho
        
    return rho

# Gradient par la méthode d'Armijo à r constant
def gradient_Armijo(x0,f,grad_f,r,PRECISION,ITER_MAX,bool):
    # Initialisation des variables
    xk = x0 # Point de départ
    k = 0   # Indice de boucle

    # Tant que le nombre d'itération max n'est pas atteint ou que le norme du gradient de la fonction est plus petit que notre précision,
    while np.linalg.norm(grad_f(xk,r)) > PRECISION and k < ITER_MAX:
        k = k + 1                             # Incrémentation de l'indice
        dk = -grad_f(xk,r)                    # Direction de descente
        rhok = Armijo(f, grad_f,xk,dk,r)      # Pas optimal
        xk = xk + rhok*dk                     # Mise à jour de xk avec sa position actuelle
        erreur = np.linalg.norm(grad_f(xk,r)) # Critère d'arrêt

    # Affichage de messages une fois la solution trouvée
    if(not bool):
        print("\nLe nombre d'itérations est k =", k)
        print("La norme du gradient en l'itéré x*, ∥∇f(x*)∥ =", np.round(erreur,5))
        print("La solution optimale obtenue est :", xk)
    else:
        return xk

# Gradient par la méthode d'Armijo à r variable et avec un critère d'arrêt en plus
def gradient_Armijo2(x0,f,grad_f,r0,PRECISION,ITER_MAX,bool):
    # Initialisation des variables
    xk = x0 # Point de départ
    k = 0   # Indice de boucle
    rk = r0 # Paramètre à réduire pour diminuer la pénalité

    # Première itération afin de rentrer dans le while
    k = k + 1                                          # Incrémentation de l'indice
    dk = -grad_f(xk,rk)                                # Direction de descente
    rhok = Armijo(f, grad_f,xk,dk,rk)                  # Pas optimal
    xk_new = xk + rhok*dk                              # Mise à jour de xk+1 avec sa position actuelle
    rk_new = rk * 0.5                                  # Mise à jour de rk+1
    erreur = np.linalg.norm(ft(xk_new,rk) - ft(xk,rk)) # Critère d'arrêt
    erreur2 = np.linalg.norm(ft(xk,rk))
    # xk = xk_new                                        # Mise à jour de xk avec sa position actuelle 
    rk = rk_new                                        # Mise à jour de rk

    # Tant que le nombre d'itération max n'est pas atteint ou que le norme du gradient de la fonction est plus petit que notre précision,
    while erreur > PRECISION*erreur2 and k < ITER_MAX and f(xk,rk) > 0:
        k = k + 1                                          # Incrémentation de l'indice
        dk = -grad_f(xk,rk)                                # Direction de descente
        rhok = Armijo(f, grad_f,xk,dk,rk)                  # Pas optimal
        xk_new = xk + rhok*dk                              # Mise à jour de xk+1 avec sa position actuelle
        rk_new = rk * 0.5                                  # Mise à jour de rk+1
        erreur = np.linalg.norm(ft(xk_new,rk) - ft(xk,rk)) # Critère d'arrêt
        erreur2 = np.linalg.norm(ft(xk,rk))
        xk = xk_new                                        # Mise à jour de xk avec sa position actuelle            
        rk = rk_new                                        # Mise à jour de rk

    # Affichage de messages une fois la solution trouvée
    if(not bool):
        print("\nLe nombre d'itérations est k =", k)
        print("La norme du gradient en l'itéré x*, ∥∇f(x*)∥ =", np.round(erreur/erreur2,5))
        print("La solution optimale obtenue est :", xk)
    else:
        return xk

# Gradient par la méthode d'Armijo à r variable
def gradient_Armijo3(x0,f,grad_f,r0,PRECISION,ITER_MAX,bool):
    # Initialisation des variables
    xk = x0 # Point de départ
    k = 0   # Indice de boucle
    rk = r0 # Paramètre à réduire pour diminuer la pénalité

    # Première itération afin de rentrer dans le while
    k = k + 1                                    # Incrémentation de l'indice
    dk = -grad_f(xk,rk)                          # Direction de descente
    rhok = Armijo(f, grad_f,xk,dk,rk)            # Pas optimal
    xk_new = xk + rhok*dk                        # Mise à jour de xk+1 avec sa position actuelle
    rk_new = rk * 0.5                            # Mise à jour de rk+1
    erreur = np.linalg.norm(ft(xk_new,rk) - ft(xk,rk)) # Critère d'arrêt
    erreur2 = np.linalg.norm(ft(xk,rk))
    # xk = xk_new                                  # Mise à jour de xk avec sa position actuelle 
    rk = rk_new                                  # Mise à jour de rk

    # Tant que le nombre d'itération max n'est pas atteint ou que le norme du gradient de la fonction est plus petit que notre précision,
    while erreur > PRECISION*erreur2 and k < ITER_MAX:
        k = k + 1                                          # Incrémentation de l'indice
        dk = -grad_f(xk,rk)                                # Direction de descente
        rhok = Armijo(f, grad_f,xk,dk,rk)                  # Pas optimal
        xk_new = xk + rhok*dk                              # Mise à jour de xk+1 avec sa position actuelle
        rk_new = rk * 0.5                                  # Mise à jour de rk+1
        erreur = np.linalg.norm(ft(xk_new,rk) - ft(xk,rk)) # Critère d'arrêt
        erreur2 = np.linalg.norm(ft(xk,rk))
        xk = xk_new                                        # Mise à jour de xk avec sa position actuelle            
        rk = rk_new                                        # Mise à jour de rk

    # Affichage de messages une fois la solution trouvée
    if(not bool):
        print("\nLe nombre d'itérations est k =", k)
        if(erreur2 != 0):
            print("La norme du gradient en l'itéré x*, ∥∇f(x*)∥ =", np.round(erreur/erreur2,5))
        else:
            print("La norme du gradient en l'itéré x*, ∥∇f(x*)∥ =")
        print("La solution optimale obtenue est :", xk)
    else:
        return xk

# Gradient par la méthode d'Armijo à r variable
def gradient_projete(x0,f,grad_f,r0,PRECISION,ITER_MAX,bool):
    # Initialisation des variables
    xk = x0 # Point de départ
    k = 0   # Indice de boucle

    # Première itération afin de rentrer dans le while
    k = k + 1                            # Incrémentation de l'indice
    dk = -grad_f(xk,r0)                  # Direction de descente
    rhok = Armijo(f, grad_f,xk,dk,r0)    # Pas optimal
    xk_new = np.maximum(xk + rhok*dk,1)  # Mise à jour de xk+1 avec sa position actuelle
    erreur = np.linalg.norm(xk_new - xk) # Critère d'arrêt
    xk = xk_new                          # Mise à jour de xk avec sa position actuelle 

    # Tant que le nombre d'itération max n'est pas atteint ou que le norme du gradient de la fonction est plus petit que notre précision,
    while erreur > PRECISION and k < ITER_MAX:
        k = k + 1                            # Incrémentation de l'indice
        dk = -grad_f(xk,r0)                  # Direction de descente
        rhok = Armijo(f, grad_f,xk,dk,r0)    # Pas optimal
        xk_new = np.maximum(xk + rhok*dk,1)  # Mise à jour de xk+1 avec sa position actuelle
        erreur = np.linalg.norm(xk_new - xk) # Critère d'arrêt
        xk = xk_new                          # Mise à jour de xk avec sa position actuelle

    # Affichage de messages une fois la solution trouvée
    if(not bool):
        print("\nLe nombre d'itérations est k =", k)
        print("La norme du gradient en l'itéré x*, ∥∇f(x*)∥ =", np.round(erreur,5))
        print("La solution optimale obtenue est :", xk)
    else:
        return xk

# Gradient par la méthode d'Armijo à r variable
def penalisation_interieure(x0,f,grad_f,r0,PRECISION,ITER_MAX,bool):
    # Initialisation des variables
    xk = x0 # Point de départ
    k = 0   # Indice de boucle
    rk = r0 # Paramètre à réduire pour diminue la pénalité

    # Première itération afin de rentrer dans le while
    k = k + 1                            # Incrémentation de l'indice
    dk = -grad_f(xk,rk)                  # Direction de descente
    rhok = Armijo(f, grad_f,xk,dk,rk)    # Pas optimal
    xk_new = xk + rhok*dk                # Mise à jour de xk+1 avec sa position actuelle
    erreur = np.linalg.norm(xk_new - xk) # Critère d'arrêt
    # xk = xk_new                          # Mise à jour de xk avec sa position actuelle 
    rk = rk * 0.5                        # Mise à jour de r

    # Tant que le nombre d'itération max n'est pas atteint ou que le norme du gradient de la fonction est plus petit que notre précision,
    while erreur > PRECISION and k < ITER_MAX:
        k = k + 1                            # Incrémentation de l'indice
        dk = -grad_f(xk,rk)                  # Direction de descente
        rhok = Armijo(f, grad_f,xk,dk,rk)    # Pas optimal
        xk_new = xk + rhok*dk                # Mise à jour de xk+1 avec sa position actuelle
        erreur = np.linalg.norm(xk_new - xk) # Critère d'arrêt
        xk = xk_new                          # Mise à jour de xk avec sa position actuelle            
        rk = rk * 0.5                        # Mise à jour de r

    # Affichage de messages une fois la solution trouvée
    if(not bool):
        print("\nLe nombre d'itérations est k =", k)
        print("La norme du gradient en l'itéré x*, ∥∇f(x*)∥ =", np.round(erreur,5))
        print("La solution optimale obtenue est :", xk)
    else:
        return xk

# Affichage en fonction de la question
def switch(case):
    if(case == 1):
        print("\nMéthode du gradient d'Armijo avec la fonction barrière inverse à r constant :")
        gradient_Armijo(x0,p,grad_p,r,PRECISION,ITER_MAX,False)
    elif(case == 2):
        print("\n∥f(x0) - f(x(0))∥ =", np.linalg.norm(ft(x0,r) - ft(gradient_Armijo(x0,p,grad_p,r,PRECISION,ITER_MAX,True),r)))
    elif(case == 3):        
        print("\nMéthode du gradient d'Armijo avec la fonction barrière inverse à r variable :")
        gradient_Armijo3(x0,p,grad_p,r,PRECISION,ITER_MAX,False)
    elif(case == 4):
        print("\nMéthode du gradient d'Armijo avec la fonction barrière logarithmique :")
        gradient_Armijo2(x0,logp,grad_logp,r,PRECISION,ITER_MAX,False)
    elif(case == 5):
        print("\nMéthode du gradient projeté :")
        gradient_projete(x0,ft,grad_ft,r,PRECISION,ITER_MAX,False)
    elif(case == 6):
        print("\nMéthode de pénalisation intérieure :")
        penalisation_interieure(x0,p,grad_p,r,PRECISION,ITER_MAX,False)
        print("\nMéthode du gradient projeté :")
        gradient_projete(x0,p,grad_p,r,PRECISION,ITER_MAX,False)
    else:
        print("\nVous n'avez saisi aucun exercice !")

# Initialisation de variables pour les tests
x0 = 2             # Point de départ
r = 3              # 
PRECISION = 10**-4 # Précision
ITER_MAX = 1000    # Nombre d'itérations max

# Affichage d'un menu
print("\n-- TD5 - Méthodes itératives d'optimisation sous contrainte --\n")
print("1 -- Exercice 1 - Question 1.b")
print("2 -- Exercice 1 - Question 1.c")
print("3 -- Exercice 1 - Question 1.d")
print("4 -- Exercice 1 - Question 2")
print("5 -- Exercice 2")
print("6 -- Exercice 4")

# Choix de l'utilisateur et affichage du choix
case = int(input("\nChoisissez une question : "))

# Affichage de messages
print("\nValeurs de test :")
print("Point de départ =", x0)
print("Précision =", PRECISION)
print("Itération maximale =", ITER_MAX)

# Appel des fonctions en fonction du choix effectué
switch(case)
print("")