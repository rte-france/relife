import numpy as np
import matplotlib.pyplot as plt
from relife.lifetime_models import Exponential
from relife.stochastic_processes import RenewalProcess

#Création d'un modèle de durée de vie
expo_model = Exponential(0.1)

#Création d'un processus de renouvellement
process_expo = RenewalProcess(lifetime_model = expo_model)

#Testons la méthode mean_age
timeline_exp, value_exp = process_expo.mean_age(tf=100, nb_steps=50) #C'est ce qu'on appelle le unpacking

#Affichons les résultats
print("Timeline", timeline_exp)
print("values", value_exp)
print("E(x) théorique", expo_model.mean()) #Affiche l'espérance mathématique théorique d'une loi exponentielle
print("la variance", expo_model.var()) #Affiche la variance théorique d'une loi exponentielle
print("fonction de survie", expo_model.sf(10)) #Affiche la fonction de survie d'une loi théorique

#Affichons le graphique
plt.plot(timeline_exp, value_exp, color="red", linewidth = 2, label= "Age moyen")
plt.xlabel("Temps t")
plt.ylabel("E[A(t)]")
plt.title("Evolution de l'âge moyen E[A(t)]")
plt.grid(True)
plt.figure(figsize = (8,5))
plt.legend()
plt.show()





    
