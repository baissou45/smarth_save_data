import json
from datasets import Dataset

# Exemple de données synthétiques avec historique des transactions
data = [
    {
        "transactions": "Dépense: 200 € (Épicerie), Revenu: 3000 € (Salaire), Dépense: 100 € (Transport)",
        "input": "J'ai un revenu mensuel de 3000 €. Comment puis-je investir ?",
        "output": "Avec un revenu de 3000 € et vos dépenses actuelles, vous pourriez épargner 500 € par mois pour investir dans un portefeuille d'actions diversifié."
    },
    {
        "transactions": "Dépense: 1500 € (Loyer), Dépense: 300 € (Nourriture), Revenu: 4000 € (Salaire)",
        "input": "Je veux acheter une maison dans 5 ans. Que dois-je faire ?",
        "output": "Avec vos dépenses actuelles, vous pourriez épargner 1000 € par mois pour un apport immobilier. Explorez également les options de prêt immobilier."
    }
]

# Convertir en format Hugging Face Dataset
dataset = Dataset.from_dict({
    "transactions": [item["transactions"] for item in data],
    "input": [item["input"] for item in data],
    "output": [item["output"] for item in data]
})

# Sauvegarder le dataset
dataset.save_to_disk("financial_chatbot_dataset")
print("Dataset généré et sauvegardé dans 'financial_chatbot_dataset'.")