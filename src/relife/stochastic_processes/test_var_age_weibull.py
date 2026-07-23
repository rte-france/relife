import numpy as np
import matplotlib.pyplot as plt
from relife.lifetime_models import Weibull
from relife.stochastic_processes import RenewalProcess

#Création d'un modèle de durée de vie
model = Weibull(2, 0.05)

#Création du processus de renouvellement
process = RenewalProcess(lifetime_model=model)

#Testons la méthode var_age
timeline,values = process.var_age(tf=100, nb_steps = 50)

#Affichons les résultats
print("Timeline", timeline)
print("Var_age", values )
print("E(X) théorique", model.mean())
print("Var théorique", model.var())
print("Fonction de survie", model.sf(10))

#Afficher le graphique
plt.figure(figsize=(8,5))
plt.plot(timeline, values, color="red", linewidth=2, label="Dispersion de l'Age")
plt.title("Evolution de la variance de lâge $Var[A(t)]$ ")
plt.xlabel("Temps t")
plt.ylabel("Var(A(t))")
plt.grid(True)
plt.legend()
plt.show()

