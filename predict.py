"""
Interactive prediction script for news category classification.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from news_classifier.data.loader import NewsDataLoader
from news_classifier.models.bilstm import BiLSTMNewsModel
from news_classifier.prediction.predictor import NewsPredictor
from news_classifier.evaluation.evaluator import ModelEvaluator
from news_classifier.utils.config import Config


def main():
    """Main prediction pipeline."""
    print("\n" + "="*60)
    print("News Category Classification - Prediction")
    print("="*60 + "\n")

    # Load tokenizer
    print("Loading tokenizer...")
    data_loader = NewsDataLoader(
        n_unique_words=Config.N_UNIQUE_WORDS,
        max_title_length=Config.MAX_TITLE_LENGTH
    )
    # Need to load data to fit tokenizer
    data_loader.load_data()
    tokenizer = data_loader.get_tokenizer()

    # Build model
    print("Building model...")
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

    # Load weights
    print("Loading trained weights...")
    evaluator = ModelEvaluator(model, output_dir=Config.OUTPUT_DIR)
    evaluator.load_weights(epoch=6)  # Load best epoch

    # Create predictor
    predictor = NewsPredictor(
        model=model,
        tokenizer=tokenizer,
        max_title_length=Config.MAX_TITLE_LENGTH,
        pad_type=Config.PAD_TYPE,
        trunc_type=Config.TRUNC_TYPE
    )

    # Run interactive mode
    predictor.interactive_mode()


if __name__ == "__main__":
    main()
