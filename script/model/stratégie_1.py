import numpy as np
import pandas as pd
from sqlalchemy import create_engine 

# Parametres de connexion
username = "root"
password = "danielmathys"
host = "127.0.0.1"
database = "data_challenge"

connection_string = f"mysql+pymysql://{username}:{password}@{host}/{database}"

# Connextion à SQLAlchemy
engine = create_engine(connection_string)


# Chargement des données
query = "SELECT * FROM initial_dataset"
initial_dataset = pd.read_sql(query, engine)

# Sélection des critères à maximiser/minimiser
dataPerf= initial_dataset[['Security ISIN' , 'Implied Temperature Rise [Â°C]','Return_1Y', 'Sharpe_1Y']]
dataPerf['Implied Temperature Rise [Â°C]']=dataPerf['Implied Temperature Rise [Â°C]'].fillna(2)

# Paramètres 
num_assets = len(dataPerf)  # Nombre d'actifs
sol_per_pop = 500  # Taille de la population
num_generations = 200  # Nombre de générations
mutation_rate = 0.1  # Taux de mutation

#Fitness value
def fitness(solution):
    portfolio_return = np.dot(solution, dataPerf['Return_1Y'])
    portfolio_sharpe = np.dot(solution, dataPerf['Sharpe_1Y'])
    
    weighted_temp_rise = np.dot(solution, dataPerf['Implied Temperature Rise [Â°C]'])
    
    if weighted_temp_rise > 2:
        return -np.inf 
    
    # Contrainte 1 : Aucun actif ne doit dépasser 10%
    penalite1 = sum((w - 0.10) ** 2 for w in solution if w > 0.10)
    
    # Contrainte 2 : La somme des actifs > 5% ne doit pas dépasser 40%
    somme_superieur_5 = sum(w for w in solution if w > 0.05)
    penalite2 = (somme_superieur_5 - 0.40) ** 2 if somme_superieur_5 > 0.40 else 0
    
    # La somme des poids doit être inférieur à 1
    penalite3 = (np.sum(solution) - 1) ** 2

    fitness_value = portfolio_return + portfolio_sharpe - (penalite1 + penalite2 + penalite3)
    return fitness_value

def select_parents(population, fitness_vals):
    num_parents = sol_per_pop // 2 
    parents = np.zeros((num_parents, num_assets))

    sorted_idx = np.argsort(fitness_vals)[::-1]
    for i in range(num_parents):
        parents[i] = population[sorted_idx[i]]

    return parents

def initialize_population():
    return np.random.dirichlet(np.ones(num_assets), size=sol_per_pop)

# Mutation
def mutate(offspring):
    for i in range(offspring.shape[0]):
        if np.random.rand() < mutation_rate:
            mutated_solution = offspring[i] + np.random.normal(0, 0.05, size=num_assets)
            mutated_solution = np.clip(mutated_solution, 0, 1)
            mutated_solution /= np.sum(mutated_solution)
            offspring[i] = mutated_solution
    return offspring

# Croisement 
def crossover(parents):
    offspring = []
    for _ in range(sol_per_pop - parents.shape[0]):
        parent1, parent2 = parents[np.random.choice(parents.shape[0], 2, replace=False)]
        crossover_point = np.random.randint(1, num_assets)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child /= np.sum(child)
        offspring.append(child)
    return np.array(offspring)

# Algorithme génétique
def genetic_algorithm():
    population = initialize_population()
    best_solution = None
    best_fitness = -np.inf
    
    for generation in range(num_generations):
        # Calcul de la fitness
        fitness_vals = np.array([fitness(sol) for sol in population])
        
        # Meilleurs parents
        parents = select_parents(population, fitness_vals)
        
        # Descendance
        offspring = crossover(parents)
        
        # Mutation
        offspring = mutate(offspring)
        
        # Nouvelle génération
        population[:parents.shape[0]] = parents
        population[parents.shape[0]:] = offspring
        
        # Meilleures solution par génération
        best_gen_solution = population[np.argmax(fitness_vals)]
        best_gen_fitness = np.max(fitness_vals)
        
        if best_gen_fitness > best_fitness:
            best_solution = best_gen_solution
            best_fitness = best_gen_fitness
        
        print(f"Generation {generation + 1} - Meilleure Fitness : {best_fitness}")
    
    return best_solution, best_fitness

best_solution, best_fitness = genetic_algorithm()

print("\nMeilleur portefeuille trouvé :")
stratégie_1 = pd.DataFrame({
    'Security ISIN': dataPerf['Security ISIN'],
    'Poids (%)': best_solution
})

print(stratégie_1) 

optimal_return = np.dot(best_solution, dataPerf['Return_1Y'])
optimal_sharpe = np.dot(best_solution, dataPerf['Sharpe_1Y'])
optimal_temp_rise = np.dot(best_solution, dataPerf['Implied Temperature Rise [Â°C]'])

print("\nMétriques du portefeuille optimal :")
print(f"Rendement du portefeuille optimal : {optimal_return:.4f}")
print(f"Ratio de Sharpe du portefeuille optimal : {optimal_sharpe:.4f}")
print(f"Température implicite moyenne du portefeuille optimal : {optimal_temp_rise:.4f}")


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

stratégie_1 = pd.merge(stratégie_1, var_viz, on = 'Security ISIN', how = 'left')

var_non_appplicable = ['Security ISIN', 'Poids (%)', 'GICS Sector' ]


for var in var_viz:
    if var not in var_non_appplicable:
        stratégie_1[var] = stratégie_1[var] * stratégie_1['Poids (%)']
        
var1 = [
    'Security ISIN', 'GICS Sector'
]
var_desc = initial_dataset[var1]

stratégie_1 = pd.merge(stratégie_1, var_desc, on = 'Security ISIN', how = 'left')

stratégie_1 = stratégie_1.rename(columns={'Poids (%)': 'Poids'})
stratégie_1 = stratégie_1.rename(columns={'GICS Sector': 'Secteur'})
stratégie_1 = stratégie_1.rename(columns={'Security ISIN': 'ISIN'})
stratégie_1 = stratégie_1.rename(columns={'Aggregated Security Climate VaR [%]': 'Climate_VaR'})

# Export vers la base de données
nom_table = 'stratégie_1'
stratégie_1.to_sql(nom_table, con=connection_string, if_exists='replace', index=False)

print("Succès !")