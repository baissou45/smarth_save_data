import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Authentifiez-vous avec votre token Hugging Face
login(token=os.environ.get("HF_TOKEN", ""))  # set HF_TOKEN env var

# Charger le modèle et le tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Exemple d'utilisation
input_text = "J'ai un revenu mensuel de 3000 €. Comment puis-je investir ?"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  # Utilisez "cpu" si vous n'avez pas de GPU
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Réponse du modèle :", response)


from transformers import MistralForCausalLM, MistralTokenizer

# Charger le modèle fine-tuné et le tokenizer
model = MistralForCausalLM.from_pretrained("./mistral-finetuned")
tokenizer = MistralTokenizer.from_pretrained("./mistral-finetuned")

# Fonction pour générer une réponse
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Tester avec une question
prompt = "Quel est mon solde de revenu pour le mois dernier ?"
response = generate_response(prompt)
print(response)
