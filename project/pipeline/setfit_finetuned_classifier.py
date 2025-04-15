import pandas as pd
import os
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset
import matplotlib.pyplot as plt

# Define the SetFitClassifier
class SetFitClassifier:
    def __init__(
        self,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        batch_size: int = 16,
        num_iterations: int = 5,
        num_epochs: int = 1
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.num_epochs = num_epochs
        self.model = None
        self.label_mapping = {"alcohol": 1, "non_alcohol": 0}
        self.reverse_label_mapping = {1: "alcohol", 0: "non_alcohol"}

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for training/prediction."""
        # Make a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Rename columns if needed
        if "ocr_text" in df_processed.columns and "text" not in df_processed.columns:
            df_processed["text"] = df_processed["ocr_text"]
        
        if "true_label" in df_processed.columns and "label" not in df_processed.columns:
            df_processed["label"] = df_processed["true_label"]
        
        # Ensure we have required columns
        if "text" not in df_processed.columns or "label" not in df_processed.columns:
            raise ValueError("DataFrame must contain 'text'/'ocr_text' and 'label'/'true_label' columns")
        
        # Convert labels to integers
        df_processed = df_processed[df_processed["label"].isin(self.label_mapping.keys())]
        df_processed["label"] = df_processed["label"].map(self.label_mapping)
        
        # Drop rows with missing values
        df_processed = df_processed.dropna(subset=["text", "label"])
        
        # Convert text column to string type to ensure all entries are strings
        df_processed["text"] = df_processed["text"].astype(str)
        
        return df_processed

    def fit_on_all(self, df: pd.DataFrame) -> None:
        """Train the model on all data."""
        # Preprocess data
        df_processed = self._preprocess_data(df)
        
        # Check class distribution
        class_counts = df_processed["label"].value_counts()
        print(f"Class distribution: {class_counts.to_dict()}")
        
        # Convert to HuggingFace Dataset
        train_dataset = Dataset.from_pandas(df_processed[["text", "label"]])
        
        # Initialize model
        self.model = SetFitModel.from_pretrained(self.model_name)
        
        # Calculate safe number of iterations
        min_class_count = min(class_counts.values)
        safe_iterations = max(1, min(self.num_iterations, min_class_count // 2))
        print(f"Using {safe_iterations} iterations for contrastive learning")
        
        # Create training arguments
        args = TrainingArguments(
            batch_size=self.batch_size,
            num_iterations=safe_iterations,
            num_epochs=self.num_epochs
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            train_dataset=train_dataset
        )
        
        # Train model
        print(f"Training SetFit model for {self.num_epochs} epochs...")
        trainer.train(args=args)
        
        # Evaluate on training data
        preds = self.model.predict(df_processed["text"].tolist())
        preds = [int(p.item() if hasattr(p, "item") else p) for p in preds]
        labels = df_processed["label"].tolist()
        
        # Calculate metrics
        accuracy = sum(p == y for p, y in zip(preds, labels)) / len(labels)
        
        # Calculate recall for each class and average
        class_recalls = {}
        for class_id in self.label_mapping.values():
            true_positives = sum(1 for p, y in zip(preds, labels) if p == class_id and y == class_id)
            actual_positives = sum(1 for y in labels if y == class_id)
            recall = true_positives / actual_positives if actual_positives > 0 else 0
            class_recalls[class_id] = recall
        
        avg_recall = sum(class_recalls.values()) / len(class_recalls)
        
        # Calculate precision for each class and average
        class_precisions = {}
        for class_id in self.label_mapping.values():
            true_positives = sum(1 for p, y in zip(preds, labels) if p == class_id and y == class_id)
            predicted_positives = sum(1 for p in preds if p == class_id)
            precision = true_positives / predicted_positives if predicted_positives > 0 else 0
            class_precisions[class_id] = precision
        
        avg_precision = sum(class_precisions.values()) / len(class_precisions)
        
        # Calculate F1 score (harmonic mean of precision and recall)
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        print(f"Training metrics:")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Avg Recall: {avg_recall:.4f}")
        print(f"  - Avg Precision: {avg_precision:.4f}")
        print(f"  - F1 Score: {f1_score:.4f}")
        
        return self
        
    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add predictions to dataframe."""
        if not self.model:
            raise ValueError("Model not trained. Call fit_on_all() first.")
        
        # Make a copy to avoid modifying the original
        results_df = df.copy()
        
        # Ensure we have text column
        if "text" not in results_df.columns and "ocr_text" in results_df.columns:
            text_column = "ocr_text"
        else:
            text_column = "text"
        
        # Make sure we have the text column
        if text_column not in results_df.columns:
            raise ValueError(f"DataFrame must contain '{text_column}' column")
        
        # Convert text column to string and handle missing values
        results_df[text_column] = results_df[text_column].fillna("").astype(str)
        
        # Make predictions
        texts = results_df[text_column].tolist()
        preds_int = self.model.predict(texts)
        
        # Convert predictions to strings
        preds_str = []
        for pred in preds_int:
            if hasattr(pred, "item"):
                pred_value = pred.item()
            else:
                pred_value = int(pred)
            preds_str.append(self.reverse_label_mapping[pred_value])
        
        # Add predictions to dataframe
        results_df["predicted"] = preds_str
        
        # Calculate accuracy if true labels are available
        if "true_label" in results_df.columns or "label" in results_df.columns:
            label_column = "true_label" if "true_label" in results_df.columns else "label"
            valid_rows = results_df[results_df[label_column].isin(self.label_mapping.keys())]
            if len(valid_rows) > 0:
                # Calculate accuracy
                accuracy = sum(valid_rows["predicted"] == valid_rows[label_column]) / len(valid_rows)
                
                # Convert labels to integers for metric calculations
                true_labels = valid_rows[label_column].map(self.label_mapping).tolist()
                pred_labels = valid_rows["predicted"].map(self.label_mapping).tolist()
                
                # Calculate recall for each class and average
                class_recalls = {}
                for class_id in self.label_mapping.values():
                    true_positives = sum(1 for p, y in zip(pred_labels, true_labels) if p == class_id and y == class_id)
                    actual_positives = sum(1 for y in true_labels if y == class_id)
                    recall = true_positives / actual_positives if actual_positives > 0 else 0
                    class_recalls[class_id] = recall
                
                avg_recall = sum(class_recalls.values()) / len(class_recalls)
                
                # Calculate precision for each class and average
                class_precisions = {}
                for class_id in self.label_mapping.values():
                    true_positives = sum(1 for p, y in zip(pred_labels, true_labels) if p == class_id and y == class_id)
                    predicted_positives = sum(1 for p in pred_labels if p == class_id)
                    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
                    class_precisions[class_id] = precision
                
                avg_precision = sum(class_precisions.values()) / len(class_precisions)
                
                # Calculate F1 score
                f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
                
                print(f"Prediction metrics:")
                print(f"  - Accuracy: {accuracy:.4f}")
                print(f"  - Avg Recall: {avg_recall:.4f}")
                print(f"  - Avg Precision: {avg_precision:.4f}")
                print(f"  - F1 Score: {f1_score:.4f}")
        
        return results_df

    def save(self, save_path):
        """Save the trained model."""
        if not self.model:
            raise ValueError("Model not trained. Call fit_on_all() first.")
        
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

    def load(self, load_path):
        """Load a trained model."""
        self.model = SetFitModel.from_pretrained(load_path)
        print(f"Model loaded from {load_path}")
        return self

# Main function to train and save the model
def main():
    # Define paths
    input_csv = "project/results/all_spell_corrected_results.csv"
    output_csv = "results/all_data_with_predictions.csv"
    model_save_path = "results/saved_model"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    print(f"Loading data from {input_csv}")
    try:
        df_raw = pd.read_csv(input_csv)
        print(f"Loaded {len(df_raw)} rows from {input_csv}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    # Initialize classifier
    clf = SetFitClassifier(
        num_epochs=1,
        batch_size=16,
        num_iterations=3
    )
    
    # Check if model already exists, if so, load it instead of retraining
    # if os.path.exists(model_save_path):
    #     print(f"Loading existing model from {model_save_path}...")
    #     clf.load(model_save_path)
    # else:
    print("Training SetFit model...")
    clf.fit_on_all(df_raw)
    clf.save(model_save_path)

    print("Making predictions...")
    # Add data type validation before prediction
    print("Checking for non-string values in text columns...")
    if "text" in df_raw.columns:
        non_str_count = sum(~df_raw["text"].apply(lambda x: isinstance(x, str)))
        if non_str_count > 0:
            print(f"Found {non_str_count} non-string values in 'text' column. Converting to strings.")
    elif "ocr_text" in df_raw.columns:
        non_str_count = sum(~df_raw["ocr_text"].apply(lambda x: isinstance(x, str)))
        if non_str_count > 0:
            print(f"Found {non_str_count} non-string values in 'ocr_text' column. Converting to strings.")
    
    df_with_preds = clf.predict_df(df_raw)

    df_with_preds.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")
    print(f"Saved model to {model_save_path}")

if __name__ == "__main__":
    main()