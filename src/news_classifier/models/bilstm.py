"""
Bidirectional LSTM model architecture for news category classification.
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, SpatialDropout1D, LSTM, Bidirectional


class BiLSTMNewsModel:
    """Bidirectional LSTM model for multi-class news classification."""

    def __init__(self, n_unique_words=20000, n_dim=128, max_title_length=70,
                 drop_embed=0.3, n_lstm=128, drop_lstm=0.3, n_classes=4):
        """
        Initialize the BiLSTM model architecture.

        Args:
            n_unique_words (int): Vocabulary size
            n_dim (int): Embedding dimension
            max_title_length (int): Input sequence length
            drop_embed (float): Dropout rate for embedding layer
            n_lstm (int): Number of LSTM units
            drop_lstm (float): Dropout rate for LSTM layer
            n_classes (int): Number of output classes (4 for AG News)
        """
        self.n_unique_words = n_unique_words
        self.n_dim = n_dim
        self.max_title_length = max_title_length
        self.drop_embed = drop_embed
        self.n_lstm = n_lstm
        self.drop_lstm = drop_lstm
        self.n_classes = n_classes
        self.model = None

    def build(self):
        """
        Build the BiLSTM model architecture.

        Returns:
            Sequential: Compiled Keras model
        """
        model = Sequential()

        # Embedding layer
        model.add(Embedding(
            self.n_unique_words,
            self.n_dim,
            input_length=self.max_title_length
        ))

        # Spatial dropout for embedding
        model.add(SpatialDropout1D(self.drop_embed))

        # Bidirectional LSTM layer
        model.add(Bidirectional(LSTM(self.n_lstm, dropout=self.drop_lstm)))

        # Output layer with softmax activation for multi-class classification
        model.add(Dense(self.n_classes, activation='softmax'))

        # Compile the model
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        self.model = model
        print("Model built successfully!")
        model.summary()

        return model

    def get_model(self):
        """
        Get the compiled model.

        Returns:
            Sequential: The Keras model
        """
        if self.model is None:
            self.build()
        return self.model
