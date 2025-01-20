import numpy as np
import pandas as pd
from sqlalchemy import create_engine 

# Parametres de connexion
username = "root"
password = "danielmathys"
host = "127.0.0.1"
database = "data_challenge"

connection_string = f"mysql+pymysql://{username}:{password}@{host}/{database}"

# Connextion Ã  SQLAlchemy
engine = create_engine(connection_string)


# Chargement des donnÃ©es
query = "SELECT * FROM initial_dataset"
initial_dataset = pd.read_sql(query, engine)

dataHybr=initial_dataset[['Security ISIN' , 'Implied Temperature Rise [Â°C]','Return_1Y', 'Sharpe_1Y', 'ESG_NEW']]
dataHybr['Implied Temperature Rise [Â°C]']=dataHybr['Implied Temperature Rise [Â°C]'].fillna(2)

# DÃ©finir les paramÃ¨tres nÃ©cessaires
num_assets = len(dataHybr)  # Nombre d'actifs dans le portefeuille
sol_per_pop = 500  # Taille de la population
num_generations = 200  # Nombre de gÃ©nÃ©rations
mutation_rate = 0.1  # Taux de mutation
# Fonction de fitness mise à jour pour maximiser uniquement le score ESG
def fitness(solution):
    # Calcul du rendement et du Sharpe
    portfolio_return = np.dot(solution, dataHybr['Return_1Y'])
    portfolio_sharpe = np.dot(solution, dataHybr['Sharpe_1Y'])
    # Calcul du score ESG moyen pondéré
    weighted_esg_score = np.dot(solution, dataHybr['ESG_NEW'])
    
    # Calcul de la température implicite moyenne pondérée
    weighted_temp_rise = np.dot(solution, dataHybr['Implied Temperature Rise [Â°C]'])

    # Pénalité pour la contrainte de température
    if weighted_temp_rise > 2:
        return -np.inf  # Pénalité sévère si la contrainte est violée
    
    # Contrainte 1 : Aucun actif ne doit dépasser 10%
    penalite1 = sum((w - 0.10) ** 2 for w in solution if w > 0.10)
    
    # Contrainte 2 : La somme des actifs > 5% ne doit pas dépasser 40%
    somme_superieur_5 = sum(w for w in solution if w > 0.05)
    penalite2 = (somme_superieur_5 - 0.40) ** 2 if somme_superieur_5 > 0.40 else 0
    
    # Pénalité pour la somme des poids différente de 1 (normalisation implicite)
    penalite3 = (np.sum(solution) - 1) ** 2

    # Fonction de fitness combinant rendement, Sharpe, et pénalités
    fitness_value = portfolio_return + portfolio_sharpe + weighted_esg_score - (penalite1 + penalite2 + penalite3)
    return fitness_value

# Sélection des meilleurs parents
def select_parents(population, fitness_vals):
    num_parents = sol_per_pop // 2  # Nombre de parents à sélectionner
    parents = np.zeros((num_parents, num_assets))

    # Trier les fitness (les meilleurs d'abord)
    sorted_idx = np.argsort(fitness_vals)[::-1]  # Ordre décroissant
    for i in range(num_parents):
        parents[i] = population[sorted_idx[i]]

    return parents

# Initialisation de la population
def initialize_population():
    return np.random.dirichlet(np.ones(num_assets), size=sol_per_pop)

# Mutation avec normalisation
def mutate(offspring):
    for i in range(offspring.shape[0]):
        if np.random.rand() < mutation_rate:
            mutated_solution = offspring[i] + np.random.normal(0, 0.05, size=num_assets)
            mutated_solution = np.clip(mutated_solution, 0, 1)
            mutated_solution /= np.sum(mutated_solution)  # Normalisation explicite
            offspring[i] = mutated_solution
    return offspring

# Croisement (crossover) avec normalisation
def crossover(parents):
    offspring = []
    for _ in range(sol_per_pop - parents.shape[0]):
        parent1, parent2 = parents[np.random.choice(parents.shape[0], 2, replace=False)]
        crossover_point = np.random.randint(1, num_assets)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child /= np.sum(child)  # Normalisation explicite
        offspring.append(child)
    return np.array(offspring)

# Algorithme génétique
def genetic_algorithm():
    population = initialize_population()
    best_solution = None
    best_fitness = -np.inf
    
    for generation in range(num_generations):
        # Calculer la fitness de la population
        fitness_vals = np.array([fitness(sol) for sol in population])
        
        # Sélectionner les meilleurs parents
        parents = select_parents(population, fitness_vals)
        
        # Créer de la descendance par croisement
        offspring = crossover(parents)
        
        # Appliquer la mutation
        offspring = mutate(offspring)
        
        # Nouvelle génération
        population[:parents.shape[0]] = parents
        population[parents.shape[0]:] = offspring
        
        # Trouver la meilleure solution de cette génération
        best_gen_solution = population[np.argmax(fitness_vals)]
        best_gen_fitness = np.max(fitness_vals)
        
        if best_gen_fitness > best_fitness:
            best_solution = best_gen_solution
            best_fitness = best_gen_fitness
        
        # Affichage des informations de la génération
        print(f"Generation {generation + 1} - Meilleure Fitness : {best_fitness}")
    
    return best_solution, best_fitness

# Exécution de l'algorithme génétique
best_solution, best_fitness = genetic_algorithm()

# Affichage du meilleur portefeuille
print("\nMeilleur portefeuille trouvé :")
stratégie3 = pd.DataFrame({
    'Security ISIN': dataHybr['Security ISIN'],
    'Poids (%)': best_solution
})


# Calculer les métriques pour le meilleur portefeuille
optimal_esg_score = np.dot(best_solution, dataHybr['ESG_NEW'])
optimal_temp_rise = np.dot(best_solution, dataHybr['Implied Temperature Rise [Â°C]'])
optimal_return = np.dot(best_solution, dataHybr['Return_1Y'])
optimal_sharpe = np.dot(best_solution, dataHybr['Sharpe_1Y'])


print("\nMétriques du portefeuille optimal :")
print(f"Rendement du portefeuille optimal : {optimal_return:.4f}")
print(f"Ratio de Sharpe du portefeuille optimal : {optimal_sharpe:.4f}")
print(f"Température implicite moyenne du portefeuille optimal : {optimal_temp_rise:.4f}")
print(f"Score ESG moyen du portefeuille optimal : {optimal_esg_score:.4f}")


variables = [
    'Security ISIN',
    'Implied Temperature Rise [Â°C]',
    'Aggregated Security Climate VaR [%]',
    'Fluvial_flooding_VaR',
    'Fluvial_flooding_cost',
    'Extreme_Heat_Climate_VaR',
    'Extreme_Heat_Cost',
    'Wildfire_Climate_VaR',
    'Wildfire_Cost',
    'Return_6M',
    'Return_1Y',
    'Return_2Y',
    'Volatility_6M',
    'Volatility_1Y',
    'Volatility_2Y',
    'Sharpe_6M',
    'Sharpe_1Y',
    'Sharpe_2Y',
    'ESG_NEW'
]

var_viz = initial_dataset[variables]

stratégie3 = pd.merge(stratégie3, var_viz, on = 'Security ISIN', how = 'left')

var_non_appplicable = ['Security ISIN', 'Poids (%)', 'GICS Sector' ]


for var in var_viz:
    if var not in var_non_appplicable:
        stratégie3[var] = stratégie3[var] * stratégie3['Poids (%)']
        
var1 = [
    'Security ISIN', 'GICS Sector'
]
var_desc = initial_dataset[var1]

stratégie3 = pd.merge(stratégie3, var_desc, on = 'Security ISIN', how = 'left')

stratégie3 = stratégie3.rename(columns={'Poids (%)': 'Poids'})
stratégie3 = stratégie3.rename(columns={'GICS Sector': 'Secteur'})
stratégie3 = stratégie3.rename(columns={'Security ISIN': 'ISIN'})
stratégie3 = stratégie3.rename(columns={'Aggregated Security Climate VaR [%]': 'Climate_VaR'})
# stratégie3 = stratégie3.rename(columns={'Fluvial Flooding Security Climate VaR [%]': 'Fluvial_flooding_VaR'})
# stratégie3 = stratégie3.rename(columns={'Fluvial Flooding Cost/Profit [USD]': 'Fluvial_flooding_VaR'})

nom_table = 'stratégie_3'
stratégie3.to_sql(nom_table, con=connection_string, if_exists='replace', index=False)

print("Succès !")