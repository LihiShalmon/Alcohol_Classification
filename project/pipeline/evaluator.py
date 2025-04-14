import wandb
import pandas as pd
import json
import numpy as np
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix,
                             classification_report)

class Evaluator:
    def __init__(self, experiment_name, text_configuration,
                 text_classification, text_correction_type):
        self.experiment_name = experiment_name
        self.text_configuration = text_configuration
        self.text_classification = text_classification
        self.text_correction_type = text_correction_type

    def evaluate(self, y_true, y_pred, test_set_name):
        metrics = self.track_all_metrics(y_true, y_pred, test_set_name)
        wandb.log(metrics)
        unique_classes = set(y_true)
        if len(unique_classes) > 1:
            try:
                wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=y_true,
                    preds=y_pred,
                    class_names=["non_alcohol", "alcohol"]
                )})
                
                report = classification_report(y_true, y_pred, output_dict=True)
                wandb.log({"classification_report": report})
            except Exception as e:
                print(f"Warning: Could not generate classification report: {e}")
        else:
            print(f"Warning: Only one class found in {test_set_name}, skipping classification report")

        print(f"Evaluation metrics logged to W&B for {self.experiment_name}")
        return metrics

    def track_all_metrics(self, y_true, y_pred, test_set_name):
        print("y_true: ", y_true)
        print("y_pred: ", y_pred)
        y_true_binary = [1 if label == "alcohol" else 0 for label in y_true]
        y_pred_binary = [1 if pred["prediction"] == "alcohol" else 0 for pred in y_pred]

        if len(set(y_true_binary)) <= 1:
            print(f"Warning: All true labels in {test_set_name} are the same class")
        
        metrics = {
            f"{test_set_name}_accuracy": accuracy_score(y_true_binary, y_pred_binary),
            f"{test_set_name}_f1": f1_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0),
            f"{test_set_name}_precision": precision_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0),
            f"{test_set_name}_recall": recall_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0),
            f"{test_set_name}_num_samples": len(y_true)
        }
        
        cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[1, 0])
        metrics[f"{test_set_name}_confusion_matrix"] = cm.tolist()
        
        # Print metrics summary
        print(f"\n{test_set_name} Metrics:")
        print(f"Accuracy: {metrics[f'{test_set_name}_accuracy']:.4f}")
        print(f"F1 Score: {metrics[f'{test_set_name}_f1']:.4f}")
        print(f"Precision: {metrics[f'{test_set_name}_precision']:.4f}")
        print(f"Recall: {metrics[f'{test_set_name}_recall']:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        return metrics

    def convert_results_to_df(self, image_paths, visual_paths,
                              texts, y_true, classifier_results_json,
                              ocr_after_correction=None, is_regex=None):
        rows = []
        
        for i in range(len(image_paths)):
            pred_result = classifier_results_json[i]
            print("pred_result: ", pred_result)
            row = {
                'image_path': image_paths[i],
                'visualization_path': visual_paths[i],
                'ocr_text': texts[i],
                'ocr_after_correction': ocr_after_correction[i] if ocr_after_correction else None,
                'true_label': y_true[i],
                'predicted_label': classifier_results_json[i]["prediction"],
                'regex_used': is_regex[i] if is_regex else False,
                "classifier_results_json": classifier_results_json[i]
            }
            
            if 'matches' in pred_result:
                row['matches'] = json.dumps(pred_result['matches'])
                row['matched_words'] = pred_result.get('matched_words', '')
                row['matched_categories'] = pred_result.get('matched_categories', '')
            else:
                row['matches'] = ''
                row['matched_words'] = ''
                row['matched_categories'] = ''
                
            rows.append(row)
            
        df = pd.DataFrame(rows)
        wandb.log({"results_table": wandb.Table(dataframe=df)})
        return df