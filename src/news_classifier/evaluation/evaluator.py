"""
Model evaluation and performance metrics for news classification.
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np
import os


class ModelEvaluator:
    """Handles model evaluation and visualization of results."""

    CATEGORIES = ['World', 'Sports', 'Business', 'Sci/Tech']

    def __init__(self, model, output_dir='model_output/news'):
        """
        Initialize the evaluator.

        Args:
            model: Trained Keras model
            output_dir (str): Directory containing saved model weights
        """
        self.model = model
        self.output_dir = output_dir

    def load_weights(self, epoch=None):
        """
        Load model weights from a specific epoch.

        Args:
            epoch (int): Epoch number to load (if None, loads the last saved epoch)
        """
        if epoch is None:
            # Find the latest checkpoint
            checkpoints = [f for f in os.listdir(self.output_dir) if f.endswith('.keras')]
            if not checkpoints:
                raise FileNotFoundError("No checkpoint files found!")
            latest = sorted(checkpoints)[-1]
            weights_path = os.path.join(self.output_dir, latest)
        else:
            weights_path = os.path.join(self.output_dir, f"weights.{epoch:02d}.keras")

        print(f"Loading weights from: {weights_path}")
        self.model.load_weights(weights_path)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model and compute metrics.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            dict: Dictionary containing evaluation metrics
        """
        print("\nEvaluating model...")

        # Get predictions
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')

        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'prediction_probs': y_pred_probs
        }

        # Print overall metrics
        print("\n" + "="*60)
        print("Overall Metrics:")
        print("="*60)
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print("="*60)

        # Print per-class metrics
        print("\nPer-Class Metrics:")
        print("="*60)
        for i, category in enumerate(self.CATEGORIES):
            TP = cm[i][i]
            FP = np.sum(cm[:, i]) - TP
            FN = np.sum(cm[i, :]) - TP

            class_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            class_recall = TP / (TP + FN) if (TP + FN) > 0 else 0

            print(f"\n{category}:")
            print(f"  Precision: {class_precision:.4f} ({class_precision*100:.2f}%)")
            print(f"  Recall:    {class_recall:.4f} ({class_recall*100:.2f}%)")

        print("="*60)

        return metrics

    def plot_confusion_matrix(self, cm, save_path=None):
        """
        Plot confusion matrix.

        Args:
            cm: Confusion matrix
            save_path (str): Path to save the plot (if None, displays plot)
        """
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.CATEGORIES)
        disp.plot(cmap='Blues', values_format='d')
        plt.title('Confusion Matrix - News Category Classification')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nConfusion matrix saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_training_history(self, history, save_path=None):
        """
        Plot training and validation accuracy/loss curves.

        Args:
            history: Training history object
            save_path (str): Path to save the plot (if None, displays plot)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss', marker='o')
        ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Model Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()
