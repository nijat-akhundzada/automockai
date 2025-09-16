
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import logging

logger = logging.getLogger(__name__)

MODEL_NAME = "distilgpt2"
MODEL_OUTPUT_DIR = "model/trained_model"


class TriplesDataset(Dataset):
    """Custom dataset for loading the (field, type, value) triples."""
    
    def __init__(self, tokenizer, file_path: str, block_size: int = 128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []

        logger.info(f"Loading and tokenizing data from {file_path}...")
        
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            logger.error(f"Training data not found at {file_path}. Please run preprocessing first.")
            raise

        # Format the data into strings
        text_data = []
        for _, row in df.iterrows():
            # Create a structured string for the model to learn from
            text = f"field: {row['field_name']} type: {row['type']} value: {row['value']}"
            text_data.append(text)
        
        # Concatenate all texts and tokenize
        full_text = "\n".join(text_data)
        tokenized_text = tokenizer.encode(full_text)
        
        # Create examples of block_size
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):
            self.examples.append(tokenized_text[i : i + block_size])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


def train_model(
    training_data_path: str,
    model_name: str = MODEL_NAME,
    output_dir: str = MODEL_OUTPUT_DIR,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
):
    """
    Fine-tunes a lightweight transformer model on the generated training data.
    """
    if not os.path.exists(training_data_path):
        logger.error(f"Training data not found at '{training_data_path}'. Please run the preprocessing script first.")
        return

    logger.info(f"Starting model training with base model: {model_name}")

    # --- 1. Initialize Tokenizer and Model ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Add a padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # Corrected: escaped backslash for pad_token value
        
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    # --- 2. Load Dataset ---
    dataset = TriplesDataset(tokenizer, file_path=training_data_path)
    
    if not dataset.examples:
        logger.error("No training examples were created. Check the training data file.")
        return

    # --- 3. Configure Training ---
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=500,
        learning_rate=learning_rate,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # --- 4. Start Training ---
    logger.info("Training in progress...")
    trainer.train()

    # --- 5. Save Model ---
    logger.info(f"Training complete. Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model saved successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # This assumes 'training_data.csv' is in the project root
    train_model("training_data.csv")
