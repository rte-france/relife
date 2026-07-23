import numpy as np
import matplotlib.pyplot as plt
from relife.lifetime_models import LogLogistic
from relife.stochastic_processes import RenewalProcess

#Création d'un modèle de durée de vie
model = LogLogistic(1, 0.05)

#Création d'un processus de renouvellement
process = RenewalProcess(lifetime_model=model)

#Testons la méthode de var_age
timeline, values = process.var_age(tf=100, nb_steps = 50)

#Affichons les résultats
print("Timeline", timeline)
print("Var[A(t)]", values)
print("Fonction de survie", model.sf(15))

#Affichons le graphique
plt.figure(figsize=(6,4))
plt.plot(timeline, values, color='blue', linewidth=2, label="Dispersion sur l'âge")
plt.title("Variance de l'âge")
plt.xlabel("Temps t")
plt.ylabel("$Var[A(t)]$")
plt.grid(True)
plt.legend()
plt.show()
