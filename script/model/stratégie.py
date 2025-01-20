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
query = "SELECT * FROM stratégie_1"
stratégie_1 = pd.read_sql(query, engine)

query = "SELECT * FROM stratégie_2"
stratégie_2 = pd.read_sql(query, engine)

query = "SELECT * FROM stratégie_3"
stratégie_3 = pd.read_sql(query, engine)

stratégie_1['stratégie'] = 1
stratégie_2['stratégie'] = 2
stratégie_3['stratégie'] = 3

stratégie = pd.concat([stratégie_1, stratégie_2, stratégie_3], ignore_index=True)

nom_table = 'stratégie'
stratégie.to_sql(nom_table, con=connection_string, if_exists='replace', index=False)

print("Succès !")