
'''
Este es el archivo donde entrenamos al modelo
y luego lo guardamos para usarlo posteriormente.

'''

#---------------Pruebas para el fine tuning-------------#
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import json

try:
    # Mensaje inicial
    print("Inicio del script")

    # Cargar el modelo pre-entrenado y el tokenizer
    model_name = "meta-llama/Meta-Llama-3-8B"
    print("Cargando el tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token="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")
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer cargado")

    print("Cargando el modelo")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token="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"
    )
    print("Modelo cargado")

    # Cargar el dataset de fine-tuning desde un archivo JSON
    print("Cargando el dataset")
    with open('dataset.json', 'r') as f:
        dataset = json.load(f)
    print("Dataset cargado")

    # Convertir el dataset a un formato compatible con Hugging Face Datasets
    print("Convirtiendo el dataset")
    dataset = Dataset.from_dict(dataset)
    print("Dataset convertido")

    # Tokenización
    def tokenize_function(examples):
        return tokenizer(examples['question'], examples['answer'], truncation=True)

    # Dataset tokenizado
    print("Tokenizando el dataset")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print("Dataset tokenizado")

    # Configurar los argumentos de entrenamiento
    print("Configurando los argumentos de entrenamiento")
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    print("Argumentos de entrenamiento configurados")

    # Crear el Trainer
    print("Creando el Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,  # Puedes tener un dataset de validación separado
    )
    print("Trainer creado")

    # Entrenar el modelo
    print("Antes Trainer.train()")
    trainer.train()
    print("Después Trainer.train()")

    # Guardar el modelo fine-tuned
    print("Guardando el modelo")
    model.save_pretrained("./modelo_fine_tuned")
    tokenizer.save_pretrained("./modelo_fine_tuned")
    print("Modelo guardado")

except Exception as e:
    print(f"Ocurrió un error en entrenamiento: {e}")
