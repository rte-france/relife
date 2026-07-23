import numpy as np
import matplotlib.pyplot as plt
from relife.lifetime_models import Gompertz
from relife.stochastic_processes import RenewalProcess

#Création du modèle de durée de vie
gompertz = Gompertz(0.1, 0.05)

#Créons le processus de renouvellement
process = RenewalProcess(lifetime_model=gompertz)

#Testons la méthode mean_age
timeline, values = process.mean_age(tf=100, nb_steps = 50)

#Affichons le résultat
print("Timeline", timeline)
print("E[A(t)]", values)
print("E(X) théorique", gompertz.mean())
print("Var(X) théorique", gompertz.var())
print("Fonction de survie", gompertz.sf(10))

#Afficher le graphique
plt.figure(figsize=(8,5))
plt.plot(timeline, values, color="red", linewidth = 2, label = "Age moyen")
plt.title("Evolution de l'âge moyen E[A(t)]")
plt.xlabel("Temps t")
plt.ylabel("E[A(t)]")
plt.grid(True)
plt.legend()
plt.show()
