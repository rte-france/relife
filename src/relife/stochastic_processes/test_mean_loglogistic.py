import numpy as np
import matplotlib.pyplot as plt
from relife.lifetime_models import LogLogistic
from relife.stochastic_processes import RenewalProcess

#Créons le modèle de durée de vie
model = LogLogistic(3, 0.5)

#Créons le processus de renouvellement
process = RenewalProcess(lifetime_model=model)

#Testons la méthode mean_age
timeline, values = process.mean_age(tf=100, nb_steps = 50)

#Affichons les résultats
print("Timeline", timeline)
print("E[A(t)]", values)
print("E[X] théorique", model.mean())
print("Var(X)", model.var())
print("fonction de survie", model.sf(10))

#Afficher le graphique
plt.figure(figsize=(8,5))
plt.plot(timeline, values, color="red", linewidth=2, label="Age moyen")
plt.title("Evolution moyen de l'actif")
plt.xlabel("Temps t")
plt.ylabel("E[A(t)]")
plt.grid(True)
plt.legend()
plt.show()
