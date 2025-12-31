"""
Evaluation script for news category classification.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from news_classifier.data.loader import NewsDataLoader
from news_classifier.models.bilstm import BiLSTMNewsModel
from news_classifier.evaluation.evaluator import ModelEvaluator
from news_classifier.utils.config import Config


def main():
    """Main evaluation pipeline."""
    print("\n" + "="*60)
    print("News Category Classification - Evaluation")
    print("="*60 + "\n")

    # Load data
    print("Loading test data...")
    data_loader = NewsDataLoader(
        n_unique_words=Config.N_UNIQUE_WORDS,
        max_title_length=Config.MAX_TITLE_LENGTH,
        pad_type=Config.PAD_TYPE,
        trunc_type=Config.TRUNC_TYPE
    )
    _, _, X_test, y_test, _ = data_loader.load_data()

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

    # Load weights and evaluate
    evaluator = ModelEvaluator(model, output_dir=Config.OUTPUT_DIR)

    # Load best weights (epoch 6 based on notebook results)
    evaluator.load_weights(epoch=6)

    # Evaluate
    metrics = evaluator.evaluate(X_test, y_test)

    # Plot confusion matrix
    print("\nGenerating confusion matrix plot...")
    evaluator.plot_confusion_matrix(
        metrics['confusion_matrix'],
        save_path=os.path.join(Config.OUTPUT_DIR, 'confusion_matrix.png')
    )

    print("\n" + "="*60)
    print("Evaluation completed!")
    print(f"Results saved in: {Config.OUTPUT_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
