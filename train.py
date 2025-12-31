"""
Main training script for news category classification.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from news_classifier.data.loader import NewsDataLoader
from news_classifier.models.bilstm import BiLSTMNewsModel
from news_classifier.training.trainer import ModelTrainer
from news_classifier.evaluation.evaluator import ModelEvaluator
from news_classifier.utils.config import Config


def main():
    """Main training pipeline."""
    print("\n" + "="*60)
    print("News Category Classification - Training")
    print("="*60 + "\n")

    # Display configuration
    Config.display()

    # Load and preprocess data
    print("Loading and preprocessing data...")
    data_loader = NewsDataLoader(
        n_unique_words=Config.N_UNIQUE_WORDS,
        max_title_length=Config.MAX_TITLE_LENGTH,
        pad_type=Config.PAD_TYPE,
        trunc_type=Config.TRUNC_TYPE
    )
    X_train, y_train, X_test, y_test, tokenizer = data_loader.load_data()

    # Build model
    print("\nBuilding model...")
    model_builder = BiLSTMNewsModel(
        n_unique_words=Config.N_UNIQUE_WORDS,
        n_dim=Config.N_DIM,
        max_title_length=Config.MAX_TITLE_LENGTH,
        drop_embed=Config.DROP_EMBED,
        n_lstm=Config.N_LSTM,
        drop_lstm=Config.DROP_LSTM,
        n_classes=Config.N_CLASSES
    )
    model = model_builder.build()

    # Train model
    print("\nInitializing training...")
    trainer = ModelTrainer(model, output_dir=Config.OUTPUT_DIR)
    history = trainer.train(
        X_train, y_train,
        X_test, y_test,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE
    )

    # Plot training history
    print("\nGenerating training history plots...")
    evaluator = ModelEvaluator(model, output_dir=Config.OUTPUT_DIR)
    evaluator.plot_training_history(
        history,
        save_path=os.path.join(Config.OUTPUT_DIR, 'training_history.png')
    )

    print("\n" + "="*60)
    print("Training completed successfully!")
    print(f"Model checkpoints saved in: {Config.OUTPUT_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
