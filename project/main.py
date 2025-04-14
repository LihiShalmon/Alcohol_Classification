import os
import pandas as pd
import config
from pipeline.ocr_processor import OCRProcessor
from pipeline.data_utils import DataUtils
from pipeline.text_classifier import TextClassifier
from pipeline.evaluator import Evaluator
from PIL import Image, ImageDraw

class PipelineRunner:
    def __init__(self, image_dir, save_dir, symspell_correction=False):
        print(f"Working from {os.curdir}")
        self.data_utils = DataUtils(image_dir)
        self.ocr = OCRProcessor(config.ocr_config, symspell_correction=symspell_correction)
        self.classifier = TextClassifier()
        self.evaluator = Evaluator(config.experiment_name, 'visual_error_corrected', 'regex_classificatoin', 'blacklist_then_whitelist')
        self.save_dir = save_dir
    def run(self):
        train, val, test = self.image_pipeline.in_distribution_splits()
        
        # Process each dataset individually
        train_metrics = self.process_dataset("all_data", train)
        # val_metrics = self.process_dataset("val", val)
        # test_metrics = self.process_dataset("test", test)
        
        # Optionally combine results or perform additional analysis
        print("\n===== Experiment Summary =====")
        print(f"Train accuracy: {train_metrics['train_set_accuracy']:.4f}")
        # print(f"Validation accuracy: {val_metrics['val_set_accuracy']:.4f}")
        # print(f"Test accuracy: {test_metrics['test_set_accuracy']:.4f}")

    def process_dataset(self, dataset_name, image_paths):
        print(f"\n===== Processing {dataset_name} dataset: {len(image_paths)} images =====")
        
        # Process images with OCR
        print(f"Running OCR on {dataset_name} images...")
        processed_data = []
        for i, path in enumerate(image_paths):
            try:
                print(f"  Processing image {i+1}/{len(image_paths)}: {os.path.basename(path)}")
                result = self.ocr.process_image(path, save_dir=self.save_dir)
                label = os.path.basename(os.path.dirname(path))
                result['label'] = label
                processed_data.append(result)
            except Exception as e:
                print(f"  Error processing image {path}: {e}")
        
        print(f"Successfully processed {len(processed_data)}/{len(image_paths)} {dataset_name} images")
        
        # Generate predictions
        print(f"Generating predictions for {dataset_name} dataset...")
        predictions = self.classifier.generate_all_regex_predictions([r['ocr_text'] for r in processed_data])
        print(f"{dataset_name} predictions: {len(predictions)}")
        
        # Evaluate results
        print(f"Evaluating {dataset_name} results...")
        image_paths = [r['image_path'] for r in processed_data]
        metrics = self.evaluator.evaluate(
            self.image_pipeline.extract_labels_from_paths(image_paths),
            predictions,
            f"{dataset_name}_set"
        )
        
        # Save results to CSV
        output_file = f"{dataset_name}.csv"
        self.save_results_of_set(processed_data, predictions, output_file)
        print(f"Saved {dataset_name} results to {output_file}  - columns = {metrics.columns}")
        return metrics
    

    def save_results_of_set(self, train_data, train_preds, file_name="results.csv"):
        df = self.evaluator.convert_results_to_df(
            [r['image_path'] for r in train_data],
            [r['ocr_visualization'] for r in train_data],
            [r['ocr_text'] for r in train_data],
            [r['label'] for r in train_data],
            train_preds, 
            ocr_after_correction=[r['ocr_text'] for r in train_data],
        )
        df.to_csv(os.path.join(self.save_dir, file_name), index=False)

def run_all_experiments():
    import wandb

    # # Experiment 1: regex_no_spelling_corrector
    # try:
    #     config.experiment_name = "regex_no_spelling_corrector"
    #     config.paths_config["save_dir"] = f"./results/{config.experiment_name}"
    #     wandb.init(project="alcohol-content-detection", name=config.experiment_name)
    #     runner = PipelineRunner(
    #         image_dir=config.paths_config["images_dir"],
    #         save_dir=config.paths_config["save_dir"],
    #         symspell_correction=False
    #     )
    #     runner.run()
    #     wandb.finish()
    # except Exception as e:
    #     print(f"Error in experiment 1: {e}")
    #     wandb.finish()

    # Experiment 2: regex_with_spell_correction
    try:
        config.experiment_name = "regex_with_spell_correction"
        config.paths_config["save_dir"] = f"./results/{config.experiment_name}"
        wandb.init(project="alcohol-content-detection", name=config.experiment_name)
        runner = PipelineRunner(
            image_dir=config.paths_config["images_dir"],
            save_dir=config.paths_config["save_dir"],
            symspell_correction=True
        )
        runner.run()
        wandb.finish()
    except Exception as e:
        print(f"Error in experiment 2: {e}")
        wandb.finish()

    # # Experiment 3: naive_ocr
    # try:
    #     config.experiment_name = "naive_ocr" ### ocr with default ocnfig
    #     config.paths_config["save_dir"] = f"./results/{config.experiment_name}"
    #     config.ocr_engine_config = {"lang": 'en', "use_gpu": False}
    #     wandb.init(project="alcohol-content-detection", name=config.experiment_name)
    #     runner = PipelineRunner(
    #         image_dir=config.paths_config["images_dir"],
    #         save_dir=config.paths_config["save_dir"],
    #         symspell_correction=True
    #     )
    #     runner.run()
    #     wandb.finish()
    # except Exception as e:
    #     print(f"Error in experiment 3: {e}")
    #     wandb.finish()

    # # Experiment 4: without_visual_correction
    # try:
    #     config.experiment_name = "without_visual_correction"
    #     config.paths_config["save_dir"] = f"./results/{config.experiment_name}"
    #     config.ocr_engine_config = {
    #         "use_angle_cls": True,
    #         "lang": 'en',
    #         "use_gpu": False,
    #         "det_db_thresh": 0.2,
    #         "det_db_box_thresh": 0.4,
    #         "det_db_unclip_ratio": 2.0
    #     }
    #     config.ocr_config = {
    #         "OCR_CHARACTER_CORRECTIONS": {"DONT USE VISUAL": "CORRECTOR"}
    #     }
    #     wandb.init(project="alcohol-content-detection", name=config.experiment_name)
    #     runner = PipelineRunner(
    #         image_dir=config.paths_config["images_dir"],
    #         save_dir=config.paths_config["save_dir"],
    #         symspell_correction=False
    #     )
    #     runner.run()
    #     wandb.finish()
    # except Exception as e:
    #     print(f"Error in experiment 4: {e}")
    #     wandb.finish()

if __name__ == "__main__":
    expected_path = os.path.join(os.getcwd(), "project", "main.py")
    print(f"Looking for file at: {expected_path}")
    print(f"File exists: {os.path.exists(expected_path)}")
    file_path =  expected_path #"./project/main.py"
    file_directory = os.path.dirname(file_path)
    os.chdir(file_directory)
    run_all_experiments()
