import numpy as np
import matplotlib.pyplot as plt
from relife.lifetime_models import Gamma
from relife.stochastic_processes import RenewalProcess


plt.figure(figsize=(8,5))
#Création d'un modèle de durée de vie
for k in [0.5, 1, 2,3,4,5] :
    model = Gamma(k, 0.05)

# création du processus de renouvellement
    process = RenewalProcess(lifetime_model=model)

#Testons la méthode mean_age
    timeline, values = process.mean_age(tf=100, nb_steps=50)

#Affichons le résultat
    print("Timeline", timeline)
    print("E[A(t)]", values)
    print("E(X) théorique", model.mean())
    print("Var(X) théorique", model.var())
    print("Fonction de survie", model.sf(10))
    plt.plot(timeline, values, linewidth=2, label=f"k={k}")

#Affichons le graphique
    
plt.title("Evolution de l'âge moyen E[A(t)]")
plt.xlabel("Temps t")
plt.ylabel("E[A(t)]")
plt.grid(True)
plt.legend()
plt.show()