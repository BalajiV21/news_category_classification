"""
Data loading and preprocessing for AG News dataset.
"""

from datasets import load_dataset
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np


class NewsDataLoader:
    """Handles loading and preprocessing of AG News dataset."""

    # News categories (labels remapped from 1-4 to 0-3)
    CATEGORIES = ['World', 'Sports', 'Business', 'Sci/Tech']

    def __init__(self, n_unique_words=20000, max_title_length=70,
                 pad_type='pre', trunc_type='pre'):
        """
        Initialize the data loader with preprocessing parameters.

        Args:
            n_unique_words (int): Maximum number of unique words to keep
            max_title_length (int): Maximum length of title sequences
            pad_type (str): Padding strategy ('pre' or 'post')
            trunc_type (str): Truncation strategy ('pre' or 'post')
        """
        self.n_unique_words = n_unique_words
        self.max_title_length = max_title_length
        self.pad_type = pad_type
        self.trunc_type = trunc_type
        self.tokenizer = None

    def load_data(self):
        """
        Load and preprocess AG News dataset.

        Returns:
            tuple: (X_train, y_train, X_test, y_test, tokenizer) - preprocessed data
        """
        print("Loading AG News dataset...")
        ds = load_dataset("sh0416/ag_news")

        # Extract titles and labels (remap from 1-4 to 0-3)
        X_train = np.array(ds['train']['title'])
        y_train = np.array(ds['train']['label']) - 1
        X_test = np.array(ds['test']['title'])
        y_test = np.array(ds['test']['label']) - 1

        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")

        # Create and fit tokenizer
        print("\nTokenizing text...")
        self.tokenizer = Tokenizer(num_words=self.n_unique_words)
        self.tokenizer.fit_on_texts(X_train)

        # Convert text to sequences
        X_train = self.tokenizer.texts_to_sequences(X_train)
        X_test = self.tokenizer.texts_to_sequences(X_test)

        # Pad sequences
        X_train = pad_sequences(
            X_train,
            maxlen=self.max_title_length,
            padding=self.pad_type,
            truncating=self.trunc_type,
            value=0
        )

        X_test = pad_sequences(
            X_test,
            maxlen=self.max_title_length,
            padding=self.pad_type,
            truncating=self.trunc_type,
            value=0
        )

        print(f"Vocabulary size: {min(len(self.tokenizer.word_index), self.n_unique_words)}")
        print(f"Sequence length: {self.max_title_length}")

        return X_train, y_train, X_test, y_test, self.tokenizer

    def get_tokenizer(self):
        """
        Get the fitted tokenizer.

        Returns:
            Tokenizer: Fitted Keras tokenizer
        """
        return self.tokenizer

    @staticmethod
    def get_category_name(label_index):
        """
        Get category name from label index.

        Args:
            label_index (int): Label index (0-3)

        Returns:
            str: Category name
        """
        return NewsDataLoader.CATEGORIES[label_index]
