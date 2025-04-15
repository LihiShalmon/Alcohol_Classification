import pandas as pd, wandb, config ,os, random
from pipeline.data_utils import DataUtils
from pipeline.regex_based_classifier import TextClassifier
from pipeline.evaluator import Evaluator
from pipeline.ocr_processor import OCRProcessor, TextCorrect
from pipeline.setfit_finetuned_classifier import SetFitClassifier
 
class PipelineRunner:

    """Main pipeline orchestrator for alcohol content detection."""
    def __init__(self, image_dir, save_dir, ocr_config, text_correction_config,
                  alcohol_keywords, experiment_name="experiment"):
        """
        Initialize pipeline with explicit configuration.
        """
        print(f"Working from {os.curdir}")
        
        # Set up directories
        self.image_dir = image_dir
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.data_utils = DataUtils(image_dir)
        text_corrector = None
        text_corrector = TextCorrect(
                use_basic_correction=text_correction_config.get("use_autocorrector", True),
                use_advanced_correction=text_correction_config.get("advanced_spell_correction", False),
                custom_substitutions=text_correction_config.get("common_ocr_mistakes"),
                dictionary_terms=alcohol_keywords
            )
            
        self.ocr_processor = OCRProcessor(ocr_config=ocr_config, text_corrector=text_corrector)
        self.classifier = TextClassifier(alcohol_keywords=alcohol_keywords)
        self.evaluator = Evaluator(  experiment_name=experiment_name)
        
    def run_all_experiments(test_images_path = "images/" ,should_re_train=False):
        experiments =config.experiments
        
        for experiment in experiments:
            try:
                exp_name = experiment["name"]
                save_dir = f"./results/{exp_name}"
                
                # Initialize W&B
                wandb.init(
                    project="alcohol-content-detection",
                    name=exp_name,
                    config=experiment
                )
                
                runner = PipelineRunner(
                    image_dir=config.paths_config["images_dir"],
                    save_dir=save_dir,
                    ocr_config=experiment["ocr_engine_settings"],
                    text_correction_config=experiment["text_correction"],
                    alcohol_keywords=config.regex_classification_terms,
                    experiment_name=exp_name
                )
                metrics = runner.run(test_images_path = "images/" ,should_re_train=False)
                
                # Finish W&B run
                wandb.finish()
                
                print(f"Experiment {exp_name} completed successfully.")
                
            except Exception as e:
                print(f"Error in experiment {experiment['name']}: {e}")
                wandb.finish()
    def run(self, test_images_path = "images/" ,should_re_train=False):
        """Execute the pipeline on training data only."""
        train, test = self.data_utils.in_distribution_splits()
        print("Processing training images with OCR...")
        if should_re_train :
            train_data = self._process_images(train)
            print(f"Predicting on training set... {[r['ocr_text'] for r in train_data][:15]}")
            train_preds = self.classifier.generate_all_regex_predictions([r['ocr_text'] for r in train_data])
            print("Training Results:")
            train_metrics = self.evaluator.evaluate(
                [r['label'] for r in train_data],
                train_preds,
                "train_set" )
            self._save_results(train_data, train_preds, "train.csv")
        else:
            # extracting ocr text from images
            test_data = self._process_images(test)
            test_preds = self.classifier.generate_all_regex_predictions([r['ocr_text'] for r in test_data])
            self.evaluator.evaluate( [r['label'] for r in test_data],  test_preds, "test_set" ) 
            self._save_results(test_data, test_preds, "test.csv")
            print(f"csv saved to {self.save_dir}/test.csv")

            # predicting over the test set 
            model_path = "results/saved_model"  # Adjust this to your actual model path
            clf = SetFitClassifier() 
            clf.load(model_path)  # Load from pretrained model path
            
            print(f"Predicting on test set {self.save_dir}/test.csv")
            test_df = pd.read_csv(f"{self.save_dir}/test.csv")
            results_df = clf.predict_df(test_df)
            results_df.to_csv(f"{self.save_dir}/test_predictions.csv", index=False)
            print(f"Predictions saved to {self.save_dir}/test_predictions.csv")

            

    def train_classify_with_setfit():
        # Train + Predict
        df_raw = pd.read_csv(input_csv_path)
        clf = SetFitClassifier()
        clf.fit_on_all(df_raw)
        df_with_preds = clf.predict_df(df_raw)

        # Save outputs
        df_with_preds.to_csv(output_csv_path, index=False)
        clf.save(model_save_path)
        print(f"\n Saved predictions to {output_csv_path}  & Model to {model_save_path}")
    def _process_images(self, image_list):
        """
        Process a list of images with OCR.
        """
        results = []
        for path in image_list:
            try:
                result = self.ocr_processor.process_image(path, save_dir=self.save_dir)
                result['label'] = os.path.basename(os.path.dirname(path))
                results.append(result)
            except Exception as e:
                print(f"Error processing image {path}: {e}")
        return results

    def _save_results(self, data, predictions, filename):
        """
        Save results to CSV file.
        """
        df = self.evaluator.convert_results_to_df(
            [r['image_path'] for r in data],
            [r['ocr_visualization'] for r in data],
            [r['ocr_text'] for r in data],
            [r['label'] for r in data],
            predictions,
            ocr_after_correction=[r['ocr_text'] for r in data]
        )
        output_path = os.path.join(self.save_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
            
if __name__ == "__main__":
    expected_path = os.path.join(os.getcwd(), "project", "main.py")    
    file_path =  expected_path #"./project/main.py"
    os.chdir( os.path.dirname(file_path))
    # #train 
    # PipelineRunner.run_all_experiments(test_images_path = "images/" ,should_re_train=True)
    
    #test 
    os.chdir( os.path.dirname(file_path))
    input_csv = "project\results\all_spell_corrected_results.csv"
    test_images_path = "images/"
    output_csv_path = "results/results.csv"
    model_save_path = "results/saved_model"
    PipelineRunner.run_all_experiments(test_images_path = test_images_path ,should_re_train=False)





