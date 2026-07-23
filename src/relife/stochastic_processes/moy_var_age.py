import numpy as np
import matplotlib.pyplot as plt
from relife.lifetime_models import Weibull
from relife.stochastic_processes import RenewalProcess

#Tracé sur le même graphe, la moyenne et la variance de l'âge.

#créons un modèle de durée de vie
model = Weibull(7.1, 0.036)

#Création d'un processus de renouvellement
process = RenewalProcess(lifetime_model=model)

#Testons sur la moyenne et variance de l'âge
timeline,values = process.mean_age(tf=300,nb_steps=300, ar = 22, a0 = 10)
_, e_values = process.var_age(tf=300, nb_steps=300, ar = 22, a0 = 10)

#Calcul de l'écart type
std_age = np.sqrt(e_values)

#Borne de l'intervalle de confiance
lower = values - std_age #borne inférieure
upper = values + std_age #borne supérieure

#Affichons les résultats
print("Timeline", timeline)
print("E[A(t)]", values)
print("Var[A(t)]", e_values)
print("Fonction de survie", model.sf(15))



#Affichons les deux courbes sur le même graphique
plt.figure(figsize=(6,4))
plt.plot(timeline, values, color="green", linewidth=2, label="E[A(t)]")
plt.fill_between(timeline,lower,upper, color="red",alpha=0.3,label="Ecart type")
plt.title("Moyenne de l'âge avec Ecart type - Weibull(2,0.05)")
plt.xlabel("Time t")
plt.ylabel("Age (years)")
plt.grid(True)
plt.legend()
plt.show()