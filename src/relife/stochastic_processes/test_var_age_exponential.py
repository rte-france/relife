import numpy as np
import matplotlib.pyplot as plt
from relife.lifetime_models import Exponential
from relife.stochastic_processes import RenewalProcess

#Création d'un modèle de durée de vie
model = Exponential(0.5)

#Création du processus de renouvellement
process = RenewalProcess(lifetime_model=model)

#Testons la méthode var_age
timeline, values = process.var_age(tf=100, nb_steps = 50)

#Affichons les résultats
print("Timeline", timeline)
print("var_age", values)
print("Fonction de survie", model.sf(10))

#Affichons le graphique
plt.figure(figsize=(8,5))
plt.plot(timeline, values, color='red', linewidth=2, label= "Dispersion de l'âge")
plt.title("Evolution de la variance de l'âge Var[A(t)]")
plt.xlabel("Temps")
plt.ylabel("Var[A(t)]")
plt.grid(True)
plt.legend()
plt.show()