import json
import pandas as pd
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# --- CONFIGURATION ---
MODEL_ID = "MirandaZhao/Finetuned_Essay_Scoring_Model_Epoch3"
DATA_FILE = "data/feedback.json"
HF_TOKEN = os.environ.get("HF_TOKEN")

def main():
    print("🚀 Starting Weekly Fine-Tuning...")

    # 1. LOAD DATA
    if not os.path.exists(DATA_FILE):
        print("⚠️ No feedback data found (data/feedback.json). Exiting.")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    df = pd.DataFrame(raw_data)
    
    # We need at least a few examples to train effectively
    if len(df) < 5:
        print(f"⚠️ Only {len(df)} new samples found. Skipping training until we have more.")
        return

    print(f"✅ Found {len(df)} verified samples for training.")

    # 2. PREPARE DATASET
    # Map the English feedback to the Model's ID format (0, 1, 2)
    # Matches your training logic: 0=不及格(Needs Imp), 1=良好(Good), 2=优秀(Excellent)
    label_map = {
        "Needs Improvement": 0,
        "Good": 1,
        "Excellent": 2
    }

    def map_labels(row):
        # Default to 1 (Good) if mapping fails, but print warning
        lbl = row.get('teacher_correction', 'Good')
        return label_map.get(lbl, 1)

    df['label'] = df.apply(map_labels, axis=1)
    df['text'] = df['essay'] # Rename for HF Trainer

    # Create Hugging Face Dataset
    train_dataset = Dataset.from_pandas(df[['text', 'label']])

    # 3. TOKENIZATION
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = train_dataset.map(tokenize_function, batched=True)

    # 4. LOAD MODEL
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=3)

    # 5. TRAINING SETUP
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,          # Retrain lightly on new data
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        save_strategy="no",          # Don't save intermediate checkpoints to save space
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    # 6. TRAIN & PUSH
    print("🏋️‍♂️ Training model...")
    trainer.train()

    print("💾 Pushing updated model to Hugging Face...")
    # This overwrites the existing model with the smarter version
    model.push_to_hub(MODEL_ID, token=HF_TOKEN)
    tokenizer.push_to_hub(MODEL_ID, token=HF_TOKEN)
    
    print("🎉 Success! The AI has learned from the teacher's feedback.")

if __name__ == "__main__":
    main()
