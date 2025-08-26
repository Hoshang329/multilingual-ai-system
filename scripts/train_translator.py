# scripts/train_translator.py

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import os

def train_translation_model():
    """
    Fine-tunes a pre-trained NLLB model for English-to-Hindi translation.
    """
    print("--- Starting Translation Model Training ---")

    # --- 1. Load and Prepare the Dataset ---
    print("Loading clean dataset from data/translation_clean.csv")
    df = pd.read_csv("data/translation_clean.csv")
    df.dropna(inplace=True)
    df = df.sample(n=20000, random_state=42)
    
    dataset = Dataset.from_pandas(df)
    train_test_split = dataset.train_test_split(test_size=0.1)
    dataset_dict = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })
    print(f"Dataset prepared with {len(dataset_dict['train'])} training examples and {len(dataset_dict['test'])} test examples.")

    # --- 2. Load Tokenizer and Model ---
    model_checkpoint = "facebook/nllb-200-distilled-600M"
    print(f"Loading tokenizer and model from {model_checkpoint}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, src_lang="eng_Latn", tgt_lang="hin_Deva")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    # --- 3. Preprocessing Function ---
    def preprocess_function(examples):
        inputs = [ex for ex in examples["english"]]
        targets = [ex for ex in examples["hindi"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs

    print("Tokenizing the dataset...")
    tokenized_datasets = dataset_dict.map(preprocess_function, batched=True)

    # --- 4. Set Up Trainer ---
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    output_dir = "models/nllb-finetuned-en-hi"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Configuring training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=32,
        optim="adafactor",
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- 5. Train the Model ---
    print("\nStarting model training... This will take a while!")
    trainer.train()
    print("Training complete.")

    # --- 6. Save the Final Model ---
    print(f"Saving the fine-tuned model to {output_dir}")
    trainer.save_model(output_dir)
    print("✅ Model saved successfully!")
    print("-" * 40)


if __name__ == "__main__":
    train_translation_model()