import pandas as pd
import os
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset
from typing import List, Union
import matplotlib.pyplot as plt


reverse_rename = {
    'text': 'ocr_text', 
    'label': 'true_label', 
    'predicted': 'predicted_label',
    'ocr_text': 'text_from_ocr', 
    'ocr_after_correction': 'corrected_ocr_results'
}

rename_dict = {
    "ocr_text": "text", 
    "true_label": "label", 
    "predicted_label": "predicted",
    "text_from_ocr": "ocr_text", 
    "corrected_ocr_results": "ocr_after_correction"
}


class SetFitClassifier:
    def __init__(
        self,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        batch_size: int = 16,
        num_iterations: int = 1,
        num_epochs: int = 1,
        plot_path="project/results/setfot_convergence_plot.png"
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.num_epochs = num_epochs
        self.model = None
        self.label_mapping = {"alcohol": 1, "non_alcohol": 0}
        self.reverse_label_mapping = {1: "alcohol", 0: "non_alcohol"}
        self.plot_path = plot_path

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})
        df = df[df["label"].isin(self.label_mapping.keys())]
        df["label"] = df["label"].map(self.label_mapping)
        df = df.dropna(subset=["text", "label"])
        return df

    def fit_on_all(self, df: pd.DataFrame) -> None:
        df_processed = self._preprocess_data(df.copy())
        class_counts = df_processed["label"].value_counts()
        print(f"Class distribution: {class_counts.to_dict()}")
        train_dataset = Dataset.from_pandas(df_processed[["text", "label"]])
        self.model = SetFitModel.from_pretrained(self.model_name)

        safe_iterations = max(1, min(self.num_iterations, min(class_counts.values) // 2))

        args = TrainingArguments(
            batch_size=self.batch_size,
            num_iterations=safe_iterations,
            num_epochs=self.num_epochs
        )

        trainer = Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=None
        )

        trainer.train(args=args)
        epoch_accuracies =[]
        for epoch in range(self.num_epochs):
            trainer.train(args=args)
            preds = self.model.predict(df_processed["text"].tolist())
            preds = [int(p.item() if hasattr(p, "item") else p) for p in preds]
            labels = df_processed["label"].tolist()
            accuracy = sum(p == y for p, y in zip(preds, labels)) / len(labels)
            epoch_accuracies.append(accuracy)
            print(f"Accuracy after epoch {epoch + 1}: {accuracy:.4f}")
        

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.model:
            raise ValueError("Model not trained. Call fit_on_all() first.")
        df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})
        df = df.dropna(subset=["text"])
        texts = df["text"].tolist()
        preds_int = self.model.predict(texts)
        preds_str = [self.reverse_label_mapping[int(p)] for p in preds_int]
        df["predicted_label"] = preds_str
        df = df.rename(columns={k: v for k, v in reverse_rename.items() if k in df.columns})
        return df

    def save(self, save_dir: str) -> None:
        if not self.model:
            raise ValueError("Model not trained.")
        self.model.save_pretrained(save_dir)

