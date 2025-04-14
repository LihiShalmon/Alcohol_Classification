import pandas as pd
import numpy as np
from setfit import SetFitModel, Trainer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from datasets import Dataset  # Import Hugging Face Dataset
from typing import List, Union, Optional
import os

class AlcoholClassifier:
    """
    A SetFit-based classifier to distinguish between alcohol and non-alcohol text.
    Uses contrastive learning and works with standard CSV files.
    """
    
    def __init__(
        self,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        batch_size: int = 16,
        num_iterations: int = 20,
        num_epochs: int = 1
    ):
        """
        Initialize the classifier.

        Args:
            model_name (str): Pretrained sentence transformer model name.
            batch_size (int): Batch size for training.
            num_iterations (int): Number of text pairs per sample for contrastive learning.
            num_epochs (int): Number of training epochs.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.num_epochs = num_epochs
        self.model = None
        self.label_mapping = {"alcohol": 1, "non_alcohol": 0}
        self.reverse_label_mapping = {1: "alcohol", 0: "non_alcohol"}

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with 'ocr_text' and 'true_label', optionally 'predicted_label'.

        Returns:
            pd.DataFrame: Preprocessed DataFrame with 'text' and 'label' columns, optionally 'predicted'.
        """
        required_columns = ["ocr_text", "true_label"]
        optional_columns = ["predicted_label"]
        selected_columns = [col for col in required_columns if col in df.columns]
        selected_columns += [col for col in optional_columns if col in df.columns]

        if not all(col in df.columns for col in required_columns):
            raise ValueError("CSV must contain 'ocr_text' and 'true_label' columns")

        df_processed = df[selected_columns].copy()

        rename_dict = {"ocr_text": "text", "true_label": "label", "predicted_label": "predicted"}
        df_processed = df_processed.rename(columns={k: v for k, v in rename_dict.items() if k in df_processed.columns})

        df_processed["label"] = df_processed["label"].map(self.label_mapping)
        df_processed = df_processed.dropna(subset=["text", "label"])

        return df_processed

    def split_train_test(
        self,
        csv_path: str,
        train_csv_path: str,
        test_csv_path: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> None:
        """
        Split the input CSV into train and test CSVs.

        Args:
            csv_path (str): Path to input CSV with 'ocr_text' and 'true_label'.
            train_csv_path (str): Path to save the train CSV.
            test_csv_path (str): Path to save the test CSV.
            test_size (float): Fraction of data to use as test set.
            random_state (int): Random seed for reproducibility.
        """
        df = pd.read_csv(csv_path)
        df_processed = self._preprocess_data(df)

        train_df, test_df = train_test_split(
            df_processed,
            test_size=test_size,
            random_state=random_state,
            stratify=df_processed["label"]
        )

        train_df["label"] = train_df["label"].map(self.reverse_label_mapping)
        test_df["label"] = test_df["label"].map(self.reverse_label_mapping)

        reverse_rename = {"text": "ocr_text", "label": "true_label", "predicted": "predicted_label"}
        train_df = train_df.rename(columns={k: v for k, v in reverse_rename.items() if k in train_df.columns})
        test_df = test_df.rename(columns={k: v for k, v in reverse_rename.items() if k in test_df.columns})
        train_df.to_csv(train_csv_path, index=False)
        test_df.to_csv(test_csv_path, index=False)
        print(f"Saved train CSV to {train_csv_path}")
        print(f"Saved test CSV to {test_csv_path}")

    def fit(self, train_csv_path: str, eval_csv_path: Optional[str] = None) -> None:
        """
        Train the model using the provided CSV file(s).

        Args:
            train_csv_path (str): Path to training CSV with 'ocr_text' and 'true_label'.
            eval_csv_path (str, optional): Path to evaluation CSV.
        """
        # Load and preprocess training data
        train_df = pd.read_csv(train_csv_path)
        train_df = self._preprocess_data(train_df)
        
        # Check if we have enough data for each class
        class_counts = train_df["label"].value_counts()
        print(f"Class distribution: {class_counts.to_dict()}")
        
        # Convert to Hugging Face Dataset
        train_dataset = Dataset.from_pandas(train_df[["text", "label"]])

        # Initialize model
        self.model = SetFitModel.from_pretrained(self.model_name)

        # Load and preprocess evaluation data if provided
        eval_dataset = None
        if eval_csv_path:
            eval_df = pd.read_csv(eval_csv_path)
            eval_df = self._preprocess_data(eval_df)
            eval_dataset = Dataset.from_pandas(eval_df[["text", "label"]])

        # Calculate appropriate number of iterations based on dataset size
        # For small datasets, we need fewer iterations to avoid index out of range
        min_class_count = min(class_counts.values)
        safe_num_iterations = max(1, min(self.num_iterations, min_class_count // 2))
        print(f"Using {safe_num_iterations} iterations for contrastive learning")
        
        # Set up training arguments
        from setfit import TrainingArguments
        
        # Create training arguments object
        args = TrainingArguments(
            batch_size=self.batch_size,
            num_iterations=safe_num_iterations,  # Use safe value to avoid index errors
            num_epochs=self.num_epochs
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if eval_dataset is not None else None,
        )
    
        # Train with arguments object
        trainer.train(args=args)
    def predict(self, texts: Union[str, List[str]]) -> List[str]:
        """
        Predict labels for input texts.

        Args:
            texts (str or List[str]): Text(s) to classify.

        Returns:
            List[str]: Predicted labels ("alcohol" or "non-alcohol").
        """
        if not self.model:
            raise ValueError("Model not trained. Call fit() first.")

        if isinstance(texts, str):
            texts = [texts]

        predictions = self.model.predict(texts)
        
        # Convert tensor outputs to integers if needed
        converted_predictions = []
        for pred in predictions:
            # Check if prediction is a tensor and convert to int if needed
            if hasattr(pred, 'item'):  # Check if it's a PyTorch tensor
                pred_value = pred.item()  # Convert tensor to Python int/float
            else:
                pred_value = int(pred)  # Ensure it's an integer for dictionary lookup
                
            converted_predictions.append(pred_value)
        
        # Map integer predictions to text labels
        return [self.reverse_label_mapping[pred] for pred in converted_predictions]
    def evaluate(self, csv_path: str) -> dict:
        """
        Evaluate the model using a CSV with 'ocr_text', 'true_label', and optionally 'predicted_label'.

        Args:
            csv_path (str): Path to evaluation CSV.

        Returns:
            dict: Metrics including accuracy, weighted F1, and classification report.
        """
        if not self.model:
            raise ValueError("Model not trained. Call fit() first.")

        eval_df = pd.read_csv(csv_path)
        eval_df_processed = self._preprocess_data(eval_df)

        texts = eval_df_processed["text"].tolist()
        true_labels = eval_df_processed["label"].tolist()
        pred_labels = self.model.predict(texts)

        report = classification_report(
            true_labels, pred_labels, target_names=["non-alcohol", "alcohol"], output_dict=True
        )
        metrics = {
            "accuracy": accuracy_score(true_labels, pred_labels),
            "weighted_f1": report["weighted avg"]["f1-score"],
            "classification_report": report
        }

        if "predicted" in eval_df_processed.columns:
            pred_labels_csv = eval_df_processed["predicted"].map(self.label_mapping).dropna()
            true_labels_csv = eval_df_processed["label"].dropna()
            if len(pred_labels_csv) == len(true_labels_csv) and len(pred_labels_csv) > 0:
                metrics["csv_predicted_accuracy"] = accuracy_score(true_labels_csv, pred_labels_csv)

        return metrics

    def save(self, save_dir: str) -> None:
        """
        Save the trained model.

        Args:
            save_dir (str): Directory to save the model.
        """
        if not self.model:
            raise ValueError("Model not trained. Call fit() first.")
        self.model.save_pretrained(save_dir)

    def load(self, load_dir: str) -> None:
        """
        Load a trained model.

        Args:
            load_dir (str): Directory containing the saved model.
        """
        self.model = SetFitModel.from_pretrained(load_dir)

# Initialize the classifier
classifier = AlcoholClassifier(
    model_name="paraphrase-MiniLM-L6-v2",
    batch_size=16,
    num_iterations=20,
    num_epochs=1
)

# Split the dataset into train and test CSVs
input_csv = r"C:\Users\LIHI\Documents\Uni\doubleverify\Alcohol_Detection_Classifier\project\results\regex_without_spelling_corrector\results_non_corrected.csv"# r"C:\Users\LIHI\Documents\Uni\doubleverify\Alcohol_Detection_Classifier\project\results\regex_with_spell_correction\results_spell_corrected.csv"
train_csv = r"C:\Users\LIHI\Documents\Uni\doubleverify\Alcohol_Detection_Classifier\project\results\regex_without_spelling_corrector\train.csv" #r"C:\Users\LIHI\Documents\Uni\doubleverify\Alcohol_Detection_Classifier\project\results\regex_with_spell_correction\train.csv"
test_csv = r"C:\Users\LIHI\Documents\Uni\doubleverify\Alcohol_Detection_Classifier\project\results\regex_without_spelling_corrector\test.csv"

classifier.split_train_test(
    csv_path=input_csv,
    train_csv_path=train_csv,
    test_csv_path=test_csv,
    test_size=0.2,
    random_state=42
)

# Train the model
classifier.fit(train_csv, eval_csv_path=test_csv)

# Evaluate on test set
metrics = classifier.evaluate(test_csv)
print(f"Accuracy: {metrics['accuracy']}")
print(f"Weighted F1: {metrics['weighted_f1']}")
print(metrics['classification_report'])

# Predict on new texts
texts = ["Beer 5% ABV", "Orange Juice"]
predictions = classifier.predict(texts)
print(predictions)  # ['alcohol', 'non-alcohol']

# Save the model
classifier.save("alcohol_classifier_model")

# Load and reuse
new_classifier = AlcoholClassifier()
new_classifier.load("alcohol_classifier_model")
print(new_classifier.predict("Wine 12%"))  # ['alcohol']