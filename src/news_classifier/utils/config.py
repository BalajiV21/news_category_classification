"""
Configuration management for the news category classifier.
"""


class Config:
    """Configuration parameters for the news category classifier."""

    # Data parameters
    N_UNIQUE_WORDS = 20000
    MAX_TITLE_LENGTH = 70
    PAD_TYPE = 'pre'
    TRUNC_TYPE = 'pre'

    # Model architecture
    N_DIM = 128
    DROP_EMBED = 0.3
    N_LSTM = 128
    DROP_LSTM = 0.3
    N_CLASSES = 4

    # Training parameters
    EPOCHS = 6
    BATCH_SIZE = 128

    # Paths
    OUTPUT_DIR = 'model_output/news'

    # Categories
    CATEGORIES = ['World', 'Sports', 'Business', 'Sci/Tech']

    @classmethod
    def display(cls):
        """Display current configuration."""
        print("\n" + "="*60)
        print("News Category Classifier - Configuration")
        print("="*60)
        print("\nData Parameters:")
        print(f"  Vocabulary size: {cls.N_UNIQUE_WORDS}")
        print(f"  Max title length: {cls.MAX_TITLE_LENGTH}")
        print(f"  Padding type: {cls.PAD_TYPE}")

        print("\nModel Architecture:")
        print(f"  Embedding dimension: {cls.N_DIM}")
        print(f"  Embedding dropout: {cls.DROP_EMBED}")
        print(f"  LSTM units: {cls.N_LSTM}")
        print(f"  LSTM dropout: {cls.DROP_LSTM}")
        print(f"  Number of classes: {cls.N_CLASSES}")

        print("\nTraining Parameters:")
        print(f"  Epochs: {cls.EPOCHS}")
        print(f"  Batch size: {cls.BATCH_SIZE}")

        print(f"\nOutput directory: {cls.OUTPUT_DIR}")
        print(f"Categories: {', '.join(cls.CATEGORIES)}")
        print("="*60 + "\n")
