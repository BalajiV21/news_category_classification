"""
Model training utilities and procedures.
"""

from keras.callbacks import ModelCheckpoint
import os


class ModelTrainer:
    """Handles model training with callbacks and checkpointing."""

    def __init__(self, model, output_dir='model_output/news'):
        """
        Initialize the trainer.

        Args:
            model: Compiled Keras model
            output_dir (str): Directory to save model checkpoints
        """
        self.model = model
        self.output_dir = output_dir
        self.history = None

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def train(self, X_train, y_train, X_test, y_test,
              epochs=6, batch_size=128):
        """
        Train the model with the given data.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training

        Returns:
            History: Training history object
        """
        # Create checkpoint callback
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(self.output_dir, "weights.{epoch:02d}.keras"),
            save_best_only=False,
            verbose=1
        )

        print(f"\nStarting training for {epochs} epochs...")
        print(f"Batch size: {batch_size}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}\n")

        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, y_test),
            callbacks=[checkpoint]
        )

        print("\nTraining completed!")

        return self.history

    def get_history(self):
        """
        Get training history.

        Returns:
            History: Training history object
        """
        return self.history
