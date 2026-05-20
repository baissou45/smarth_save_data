from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import json

# Charger le dataset JSON
def load_data(file_path="financial_dialogues.json"):
    with open(file_path, 'r', encoding="utf-8") as f:
        dialogues = json.load(f)
    return dialogues

# Créer un Dataset Hugging Face à partir des dialogues
def create_dataset(dialogues):
    data = {
        'input': [d['input'] for d in dialogues],
        'output': [d['output'] for d in dialogues]
    }
    return Dataset.from_dict(data)

# Tokenizer les entrées et sorties
def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples['input'],
        padding="max_length",
        truncation=True,
        max_length=512
    )

# Fonction principale
def main():
    # Charger les dialogues
    dialogues = load_data()

    # Créer le dataset
    dataset = create_dataset(dialogues)

    # Diviser en ensembles d'entraînement et de test (80% - 20%)
    dataset_split = dataset.train_test_split(test_size=0.2, shuffle=True)

    # Charger le tokenizer GPT-J
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.pad_token = tokenizer.eos_token  # Définir le pad_token comme eos_token

    # Appliquer la tokenisation sur les datasets
    tokenized_datasets = dataset_split.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["input", "output"]  # Évite les erreurs liées aux colonnes non utilisées
    )

    # Vérifier la structure du dataset
    print(tokenized_datasets)

    # Charger le modèle GPT-J
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

    # Définir les arguments d'entraînement
    training_args = TrainingArguments(
        output_dir='./results',          # Dossier de sortie
        num_train_epochs=3,              # Nombre d'époques
        per_device_train_batch_size=2,   # Taille du batch
        per_device_eval_batch_size=2,    # Batch pour l'évaluation
        warmup_steps=500,                # Nombre de pas pour le warmup
        weight_decay=0.01,               # Décroissance du poids
        logging_dir='./logs',            # Dossier de logs
        logging_steps=10,                # Fréquence des logs
        evaluation_strategy="epoch",     # Évaluation après chaque époque
        save_strategy="epoch",           # Sauvegarde après chaque époque
        save_total_limit=2,              # Limite le nombre de sauvegardes
        push_to_hub=False,               # Désactiver le push automatique vers Hugging Face
    )

    # Créer l'entraîneur
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
    )

    # Lancer l'entraînement
    trainer.train()

    # Sauvegarder le modèle fine-tuné et le tokenizer
    model.save_pretrained("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")

    print("✅ Modèle et tokenizer sauvegardés avec succès.")

# Exécuter la fonction principale
if __name__ == '__main__':
    main()
