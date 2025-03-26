## Data manipulation
from datasets import Dataset
import pandas as pd
import random

## Model training
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData
)
from sentence_transformers.losses import DenoisingAutoEncoderLoss

## Callbacks and Monitoring
import wandb

WB_LOGIN_KEY = 'Add your wandb API key here'


# Log in to wandb with the API key
wandb.login(key=WB_LOGIN_KEY, anonymous='allow')



def load_data(file_path='cluster_csv/clustering_results.csv'):
    """Load text data for denoising"""
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    return df['text'].tolist()

def apply_noise(texts, del_ratio=0.6):
    """Add noise to text by randomly deleting words"""
    import nltk
    nltk.download('punkt')
    from nltk import word_tokenize
    from nltk.tokenize.treebank import TreebankWordDetokenizer

    noisy_texts = []
    for text in texts:
        words = word_tokenize(text, language="german")
        if len(words) == 0:
            noisy_texts.append(text)
            continue
            
        # Randomly delete words
        kept_words = [word for word in words if random.random() > del_ratio]
        if len(kept_words) == 0:  # Ensure at least one word remains
            kept_words = [random.choice(words)]
            
        noisy_text = TreebankWordDetokenizer().detokenize(kept_words)
        noisy_texts.append(noisy_text)
    
    return noisy_texts

def create_datasets(texts):
    """Create noisy and clean datasets for TSDAE"""
    # Apply noise to create corrupted versions
    noisy_texts = apply_noise(texts)
    
    # Create dataset with original and noisy texts
    dataset = Dataset.from_dict({
        "text": texts,
        "noisy": noisy_texts
    })
    
    # Split into train/validation
    train_val = dataset.train_test_split(train_size=0.8, shuffle=True, seed=42)
    return train_val

if __name__ == "__main__":
    # 1. Load text data
    print("Loading data...")
    texts = load_data()
    
    # 2. Create noisy datasets
    print("Creating noisy datasets...")
    datasets = create_datasets(texts)
    
    # 3. Initialize model (keep your original config)
    model = SentenceTransformer(
        "sentence-transformers/all-distilroberta-v1",
        model_card_data=SentenceTransformerModelCardData(
            language="de",
            license="apache-2.0",
            model_name="Roberta-base fine-tuned with TSDAE"
        )
    )
    
    # 4. Define TSDAE loss
    loss = DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)
    
    # 5. Define training arguments (keep your original args)
    args = SentenceTransformerTrainingArguments(
        output_dir="models/german-tsdae",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=100,
        run_name="german-tsdae"
    )
    
    # 6. Create trainer (no evaluator needed for TSDAE)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        loss=loss
    )
    
    # 7. Train
    print("\nTraining model...")
    trainer.train()
    
    # 8. Save the model
    print("\nSaving model...")
    model.save_pretrained("models/german-tsdae/final")
    
    print("\nTraining complete!")
    wandb.finish()