##Data manipulation
from datasets import Dataset
import pandas as pd
import numpy as np
import random

##Model training
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator

##Callbacks and Monitoring
import wandb
WB_LOGIN_KEY = 'Add your wandb API key here' 


# Log in to wandb with the API key
wandb.login(key=WB_LOGIN_KEY, anonymous='allow')

def load_cluster_data(file_path='cluster_csv/clustering_results.csv'):
    """Load clustering results and extract data for best performing cluster (HDBSCAN UMAP)"""
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    return df['text'].tolist(), df['hdbscan_10_umap'].tolist()

def create_triplets(texts, cluster_labels, num_triplets_per_text=3):
    """Create triplets (anchor, positive, negative) from clustering results"""
    triplets = []
    texts = np.array(texts)
    cluster_labels = np.array(cluster_labels)
    
    # Create dictionary mapping cluster labels to text indices
    cluster_to_texts = {}
    for i, label in enumerate(cluster_labels):
        if label != -1:  # Ignore noise points
            if label not in cluster_to_texts:
                cluster_to_texts[label] = []
            cluster_to_texts[label].append(i)
    
    # Create triplets
    for idx, (text, label) in enumerate(zip(texts, cluster_labels)):
        if label == -1:  # Skip noise points
            continue
            
        # Get positive examples (same cluster)
        same_cluster_indices = cluster_to_texts[label]
        
        # Get negative examples (different clusters)
        different_cluster_indices = []
        for other_label, indices in cluster_to_texts.items():
            if other_label != label:
                different_cluster_indices.extend(indices)
        
        # Create triplets
        for _ in range(num_triplets_per_text):
            if len(same_cluster_indices) > 1 and different_cluster_indices:
                pos_idx = random.choice([i for i in same_cluster_indices if i != idx])
                neg_idx = random.choice(different_cluster_indices)
                
                triplets.append({
                    'anchor': text,
                    'positive': texts[pos_idx],
                    'negative': texts[neg_idx]
                })
    
    return triplets

def create_datasets(triplets, train_size=0.8):
    """Create train and validation datasets"""
    # Convert triplets to Dataset format
    dataset = Dataset.from_list(triplets)
    
    # Split dataset into train and validation
    train_val = dataset.train_test_split(
        train_size=train_size,
        shuffle=True,
        seed=42
    )
    
    return {
        'train': train_val['train'],
        'validation': train_val['test']
    }

if __name__ == "__main__":
    # 1. Load clustered data
    print("Loading cluster data...")
    texts, cluster_labels = load_cluster_data()
    
    # 2. Create triplets
    print("Creating triplets...")
    triplets = create_triplets(texts, cluster_labels)
    print(f"Created {len(triplets)} triplets")
    
    # 3. Create datasets
    print("Creating datasets...")
    datasets = create_datasets(triplets)
    
    # 4. Initialize model
    model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2",
        model_card_data=SentenceTransformerModelCardData(
            language="de",
            license="apache-2.0",
            model_name="MPNet-base finetuned on German production data"
        )
    )
    
    # 5. Define loss function
    loss = MultipleNegativesRankingLoss(model)


    
    # 6. Define training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir="models/all-mpnet-base-v2",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        run_name="MPNet-base finetuned"
    )
    
    # 7. Create evaluator
    evaluator = TripletEvaluator(
        anchors=datasets['validation']['anchor'],
        positives=datasets['validation']['positive'],
        negatives=datasets['validation']['negative'],
        name="validation"
    )
    
    # 8. Evaluate base model
    print("\nEvaluating base model...")
    evaluator(model)
    
    # 9. Create trainer and train
    print("\nTraining model...")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        loss=loss,
        evaluator=evaluator
    )
    trainer.train()
    
    # 10. Save the model
    print("\nSaving model...")
    model.save_pretrained("models/all-mpnet-base-v2/final-MNRL-80-20")
    
    print("\nTraining complete!")

    wandb.finish()
