import numpy as np
import matplotlib.pyplot as plt
from relife.lifetime_models import Gamma
from relife.stochastic_processes import RenewalProcess

#Création d'un modèle de durée de vie
model = Gamma(2,0.05)

#Création d'un processus de renouvellement
process = RenewalProcess(lifetime_model=model)

#Testons la méthode var_age
timeline, values = process.var_age(tf=100, nb_steps=50)

#Affichons les résultats
print("Timeline", timeline)
print("Var[A(t)]", values)
print("Fonction de survie", model.sf(10))

#Affcihage du graphique
plt.figure(figsize=(8,5))
plt.plot(timeline, values, color='red', linewidth=2, label="Dispersion sur l'âge")
plt.title("Variance de l'âge via la loi gamma")
plt.xlabel("Temps")
plt.ylabel("Var[A(t)]")
plt.grid(True)
plt.legend()
plt.show()
