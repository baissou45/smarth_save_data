import pandas as pd
import random
import json
from datetime import datetime

# Supposons que 'df' est ton DataFrame avec l'historique des transactions.
# Exemple de DataFrame :
# df = pd.read_csv('transactions.csv')  # Le fichier CSV des transactions des utilisateurs.

# Pour l'exemple, je vais créer un DataFrame fictif
data = {
    "user_id": [1, 1, 1, 2, 2],
    "institution_id": [101, 101, 102, 103, 104],
    "categorie_id": [5, 6, 7, 5, 8],
    "type": ['debit', 'credit', 'debit', 'credit', 'debit'],
    "montant": [200, 500, 150, 300, 100],
    "devise": ['USD', 'USD', 'USD', 'EUR', 'USD'],
    "date_emission": [
        '2024-01-01', '2024-01-05', '2024-01-10', '2024-01-15', '2024-01-20'
    ]
}

df = pd.DataFrame(data)
df['date_emission'] = pd.to_datetime(df['date_emission'])  # Convertir en datetime

# Créer une colonne "Entrées" et "Dépenses" selon le type de la transaction
df['Entrées'] = df.apply(lambda row: row['montant'] if row['type'] == 'credit' else 0, axis=1)
df['Dépenses'] = df.apply(lambda row: row['montant'] if row['type'] == 'debit' else 0, axis=1)

# Agréger les données par utilisateur et par mois
df['Mois'] = df['date_emission'].dt.to_period('M')
df_user_agg = df.groupby(['user_id', 'Mois']).agg({'Entrées': 'sum', 'Dépenses': 'sum'}).reset_index()

# Créer un dataset de dialogues basé sur l'historique des transactions
dialogues = []

for _, row in df_user_agg.iterrows():
    user_id = row['user_id']
    total_entrees = row['Entrées']
    total_depenses = row['Dépenses']
    mois = str(row['Mois'])
    
    # Scénarios : générer des questions et des réponses
    question_entrees = f"Quelles ont été mes entrées pour le mois {mois} ?"
    answer_entrees = f"Pour le mois {mois}, vos entrées s'élèvent à {total_entrees} USD."
    
    question_depenses = f"Quelles ont été mes dépenses pour le mois {mois} ?"
    answer_depenses = f"Pour le mois {mois}, vos dépenses s'élèvent à {total_depenses} USD."

    # Ajout de dialogues dans la liste
    dialogues.append({"input": question_entrees, "output": answer_entrees})
    dialogues.append({"input": question_depenses, "output": answer_depenses})

# Sauvegarder les dialogues dans un fichier JSON pour fine-tuning
with open('financial_dialogues.json', 'w') as f:
    json.dump(dialogues, f, indent=4)

print(f"Le fichier 'financial_dialogues.json' a été généré avec {len(dialogues)} dialogues.")
