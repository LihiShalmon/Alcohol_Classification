import os
import pandas as pd
import config
from pipeline.ocr_processor import OCRProcessor
from pipeline.image_pipeline import ImagePipeline
from pipeline.text_classifier import TextClassifier
from pipeline.evaluator import Evaluator
from PIL import Image, ImageDraw

class PipelineRunner:
    def __init__(self, image_dir, save_dir, advanced_spell_correction=False):
        print(f"Working from {os.curdir}")
        self.image_pipeline = ImagePipeline(image_dir)
        self.ocr = OCRProcessor(config.ocr_config, advanced_spell_correction=advanced_spell_correction)
        self.classifier = TextClassifier(config.regex_classification_terms,advanced_spell_correction=advanced_spell_correction)
        self.evaluator = Evaluator(config.experiment_name, 'visual_error_corrected', 'regex_classificatoin', 'blacklist_then_whitelist')
        self.save_dir = save_dir

    def run(self):
        train, val, test = self.image_pipeline.in_distribution_splits()

        def _process_images(image_list):
            results = []  # list[dict]
            for path in image_list:
                result = self.ocr.process_image(path, save_dir=self.save_dir)
                label = os.path.basename(os.path.dirname(path))
                result['label'] = label
                results.append(result)
            return results
            # list of {'image_path', 'ocr_text', 'ocr_visualization', 'label'}

        train_data = _process_images(train)
        val_data = _process_images(val)
        test_data = _process_images(test)

        # prediction on validation and test sets
        print(f"Predicting on training set... {[r['ocr_text'] for r in train_data][:10]}")
        train_preds = self.classifier.generate_all_regex_predictions([r['ocr_text'] for r in train_data])
        val_preds = self.classifier.generate_all_regex_predictions([r['ocr_text'] for r in val_data])
        test_preds = self.classifier.generate_all_regex_predictions([r['ocr_text'] for r in test_data])

        print("Training Results:")
        metrics =self.evaluator.evaluate(self.image_pipeline.extract_labels_from_paths(train),
                                 train_preds,"train_set")

        print("Validation Results:")
        metrics = self.evaluator.evaluate(self.image_pipeline.extract_labels_from_paths([r['image_path'] for r in val_data]),
                                    val_preds, "val_set")
        print("Test Results:")
        metrics = self.evaluator.evaluate(self.image_pipeline.extract_labels_from_paths([r['image_path'] for r in test_data]),
                                    test_preds, "test_set")

        self.save_results_of_set(train_data, train_preds, "train.csv")
        self.save_results_of_set(val_data, val_preds, "val.csv")
        self.save_results_of_set(test_data, test_preds, "test.csv")

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
    #         advanced_spell_correction=False
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
            advanced_spell_correction=True
        )
        runner.run()
        wandb.finish()
    except Exception as e:
        print(f"Error in experiment 2: {e}")
        wandb.finish()

    # # Experiment 3: naive_ocr
    # try:
    #     config.experiment_name = "naive_ocr"
    #     config.paths_config["save_dir"] = f"./results/{config.experiment_name}"
    #     config.ocr_engine_config = {"lang": 'en', "use_gpu": False}
    #     wandb.init(project="alcohol-content-detection", name=config.experiment_name)
    #     runner = PipelineRunner(
    #         image_dir=config.paths_config["images_dir"],
    #         save_dir=config.paths_config["save_dir"],
    #         advanced_spell_correction=True
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
    #         "use_autocorrector": True,
    #         "COMMON_OCR_MISTAKES": {"DONT": "DONT"}
    #     }
    #     wandb.init(project="alcohol-content-detection", name=config.experiment_name)
    #     runner = PipelineRunner(
    #         image_dir=config.paths_config["images_dir"],
    #         save_dir=config.paths_config["save_dir"],
    #         advanced_spell_correction=True
    #     )
    #     runner.run()
    #     wandb.finish()
    # except Exception as e:
    #     print(f"Error in experiment 4: {e}")
    #     wandb.finish()

if __name__ == "__main__":
    file_path = "./project/main.py"
    file_directory = os.path.dirname(file_path)
    os.chdir(file_directory)
    run_all_experiments()

#     # Initialize and run the pipeline
#     runner = PipelineRunner(
#         image_dir=config.paths_config["images_dir"],
#         save_dir=config.paths_config["save_dir"],
#         advanced_spell_correction=False
#     )
#     runner.run()

#     experiment_name =  "regex_with_spell_correction"
#     config.experiment_name="regex_with_spell_correction"
#     config.paths_config = {
#     # directories
#     "images_dir": "./images",
#     "alcohol_images_dir": "./images/alcohol",
#     "non_alcohol_images_dir": "./images/non_alcohol",
#     "save_dir": f"./results/{experiment_name}",
#     "results_file": "model_outputs.csv",
# }
#     # Initialize and run the pipeline
#     new_run = PipelineRunner(
#         image_dir=config.paths_config["images_dir"],
#         save_dir=config.paths_config["save_dir"],
#         advanced_spell_correction=True    )
#     new_run.run()


#     experiment_name =  "naive_ocr"
#     config.experiment_name="naive_ocr"

#     config.ocr_engine_config = { "lang" :'en',
#                                     "use_gpu":False,
#                                     }

#     new_run = PipelineRunner(
#             image_dir=config.paths_config["images_dir"],
#             save_dir=config.paths_config["save_dir"],
#             advanced_spell_correction=True    )
#     new_run.run()


#     experiment_name =  "without_visual_correction"
#     config.experiment_name="without_visual_correction"

#     config.ocr_engine_config  = {"use_angle_cls": True,
#                             "lang" :'en',
#                                     "use_gpu":False,
#                                     "det_db_thresh":0.2,
#                                     "det_db_box_thresh":0.4,
#                                     "det_db_unclip_ratio":2.0}

#     config.ocr_config = {
#         "use_autocorrector": True,
#         "COMMON_OCR_MISTAKES": {
#         "DONT": "DONT"
#         }}
#     new_run = PipelineRunner(
#             image_dir=config.paths_config["images_dir"],
#             save_dir=config.paths_config["save_dir"],
#             advanced_spell_correction=True    )
#     new_run.run()
