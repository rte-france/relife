import numpy as np
import matplotlib.pyplot as plt
from relife.lifetime_models import Gompertz
from relife.stochastic_processes import RenewalProcess

#Création d'un modèle de durée de vie
model = Gompertz(0.1, 0.05)

#Création du processus de renouvellement
process = RenewalProcess(lifetime_model=model)

#Testons la méthode sur la var_age
timeline, values = process.var_age(tf=100, nb_steps=50)

#Affichons le résultat
print("Timeline", timeline)
print("Var[A(t)]", values)
print("Fonction de survie", model.sf(20))

#Affichons le graphique
plt.figure(figsize=(6,4))
plt.plot(timeline, values, color='red', linewidth=2, label="Dispersion sur l'âge")
plt.title("Variance de l'âge")
plt.xlabel("Temps")
plt.ylabel("Var[A(t)]")
plt.grid(True)
plt.legend()
plt.show()