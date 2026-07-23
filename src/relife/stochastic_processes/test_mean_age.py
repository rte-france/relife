import numpy as np
import matplotlib.pyplot as plt
from relife.lifetime_models import Weibull
from relife.stochastic_processes import RenewalProcess

#Créons un modèle de durée de vie
model = Weibull(2, 0.05)

#Créons le processus de renouvellement
process = RenewalProcess(lifetime_model= model)

#Testons la méthode mean_age
timeline, values = process.mean_age(tf=40, nb_steps = 50)

print("Timeline", timeline)
print("E[A(t)]", values)
print("E(X) théorique", model.mean()) #pour caluler l'espérance thorique de la loi de weibull
print("Variance", model.var()) #pour calculer la variance d'une loi de weibull
print("Fonction de survie", model.sf(20)) #La fonction de survie dépend du temps

#Afficher le graphique
plt.figure(figsize=(8,5)) #pour augmenter légèrement la taille de la figure
plt.plot(timeline, values, color = "red" , linewidth = 2,  label="Age moyen")
plt.title("Evolution de l'âge moyen E[A(t)]")
plt.xlabel("Temps t")
plt.ylabel("E[A(t)]")
plt.grid(True)
plt.legend()
plt.show()
