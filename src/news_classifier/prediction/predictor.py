"""
Custom prediction utilities for news category classification.
"""

from keras.preprocessing.sequence import pad_sequences
import numpy as np


class NewsPredictor:
    """Handles custom text prediction for news category classification."""

    CATEGORIES = ['World', 'Sports', 'Business', 'Sci/Tech']

    def __init__(self, model, tokenizer, max_title_length=70,
                 pad_type='pre', trunc_type='pre'):
        """
        Initialize the predictor.

        Args:
            model: Trained Keras model
            tokenizer: Fitted Keras tokenizer
            max_title_length (int): Maximum sequence length
            pad_type (str): Padding strategy
            trunc_type (str): Truncation strategy
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_title_length = max_title_length
        self.pad_type = pad_type
        self.trunc_type = trunc_type

    def encode_text(self, text):
        """
        Encode raw text to sequence format.

        Args:
            text (str): Raw news title text

        Returns:
            np.array: Encoded and padded sequence
        """
        seq = self.tokenizer.texts_to_sequences([text])
        encoded = pad_sequences(
            seq,
            maxlen=self.max_title_length,
            padding=self.pad_type,
            truncating=self.trunc_type,
            value=0
        )
        return encoded

    def predict(self, text, verbose=True):
        """
        Predict news category for a given title.

        Args:
            text (str): News title to classify
            verbose (bool): Whether to print results

        Returns:
            tuple: (category, confidence, all_scores) - predicted category, confidence, and all class scores
        """
        encoded = self.encode_text(text)
        scores = self.model.predict(encoded, verbose=0)[0]

        pred_idx = np.argmax(scores)
        category = self.CATEGORIES[pred_idx]
        confidence = scores[pred_idx]

        if verbose:
            print(f"\nInput: {text}")
            print(f"\nPredicted Category: {category}")
            print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            print("\nAll Category Scores:")
            for i, cat in enumerate(self.CATEGORIES):
                print(f"  {cat:12s}: {scores[i]:.4f} ({scores[i]*100:.2f}%)")

        return category, confidence, scores

    def interactive_mode(self):
        """
        Run interactive prediction mode where users can input news titles.
        """
        print("\n" + "="*60)
        print("News Category Predictor - Interactive Mode")
        print("="*60)
        print("Enter news titles to classify into categories:")
        print("  - World")
        print("  - Sports")
        print("  - Business")
        print("  - Sci/Tech")
        print("\nType 'q' or 'quit' to exit.\n")

        while True:
            text = input("Enter news title: ").strip()

            if text.lower() in ['q', 'quit']:
                print("Exiting interactive mode. Goodbye!")
                break

            if text == "":
                print("Please enter a title.\n")
                continue

            self.predict(text)
            print()
