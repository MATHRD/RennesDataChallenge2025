Lauréat du prix Innovation du Rennes Data Challenge 2025 organisé par la faculté d'économie de l'Univeristé de Rennes

# Sujet
**APICIL Asset Management**

**Contruction d'un portefeuille efficient sous une trajectoire de 2°C**

Notre approche du sujet était de combiner innovation, performance et réalité. 
Nous avons accentué notre projet de façon à coller avec une problématique pouvant être retrouvé dans le monde de la finance. 
Nous avons ainsi souhaité construire plusieurs portefeuilles dans l'optique de reproduire les différents profils d'investisseur possibles.
A cela nous avons réformer notre vision des critères ESG (et plus particulièrement l'aspect environemental).
Enfin, en ouverture, grâce aux données multiples et de qualités présentes dans les différents dataset, nous avons réaliser des "Stress Tests" sur nos portefeuilles afin de comprendre comment réagissent nos actifs selon différntes catastrophes naturelles.


**Contraintes de base du sujet:**
- Aucun actif du portefeuille peut avoir un poids supérieur à 10% du portefeuille.
- La somme des poids des actifs supérieur à 5% ne doit pas dépasser 40% du poids du portefeuille.
- L'Implied Temprature Rise (ITR) du portefeuille ne doit pas dépasser les 2°degrés.

Nous avons créée 3 portefeuilles répondant à 3 diffents types d'investisseur:

- Portefeuille Performance
Contraintes: maximiser le rendement et le ratio de sharpe

- Portefeuille Green
Contrainte: minimiser notre variable ESG créé

- Portefeuille Ajusté
Containtes: maximiser le rendement et minimiser notre variable ESG


Une fois nos contraintes énumérés, nous avons pu construire un algorythme génétique modulable pour répondre aux différences entre nos 3 portefeuilles.
- [Performance](https://github.com/MATHRD/RennesDataChallenge2025/blob/main/script/model/strat%C3%A9gie_1.py)
- [Green](https://github.com/MATHRD/RennesDataChallenge2025/blob/main/script/model/strat%C3%A9gie_2.py)
- [Ajusté](https://github.com/MATHRD/RennesDataChallenge2025/blob/main/script/model/strat%C3%A9gie_3.py) 

Le modèle a pour output un vecteur avec le poids attribués à chacun des actifs de l'univers d'investissement. Une fois ce vecteur récupéré, nous pouvons multiplier ce poids avec les différents indicateurs que nous souhaitons manipuler (Rendement, Volatilité, Ratio de Sharpe, ITR, note ESG).


Concernant la construction d'un nouvel indicateur, il se basait sur des critères de pollution directe et d'investissement à la transition.
Pollution directe:
- Émission de carbone projetées (t/CO2)
- Score d'émission carbone
- Présence de l'entreprise dans un secteur énergie fossile

Critères d'investissement à la transition:
- Objectif de température
- Critère ESG rebalancé selon le secteur
- Implication dans son score ESG (amélioration de son score ESG sur la période)
- Initiative à la réduction de carbone


En dernier lieu, grâce aux données disponibles, nous avons simuler des catastrohes naturelles sur nos différents portefeuilles afin de mesurer la perte monétaire et la variation de la VaR si des catastrophes tel que des inondations, séchesseress ou incendies affectait des entreprises de notre portefeuille.

Notre projet détaillé se trouve dans le pdf 
- [Rapport](https://github.com/MATHRD/RennesDataChallenge2025/blob/main/Rapport.pdf)

Notre rendu se fait avec un Grafana accesible çi-dessous:
http://localhost:3000/dashboard/snapshot/TDUo0uSUA3Eocxcbw0sHiYcXu3rkH1up

