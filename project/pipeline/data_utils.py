import os
import random
from sklearn.model_selection import train_test_split

class DataUtils:
    def __init__(self, image_dir, test_size=0.8, val_size=0.15, seed=42):
        self.image_dir = image_dir
        self.seed = seed
        self.test_size = test_size
        self.val_size = val_size

        self.train_paths = None
        self.val_paths = None
        self.test_paths = None

    def load_image_paths(self):
        directory = self.image_dir
        image_extensions = {'.jpg', '.png'}  # Add more extensions if needed
        image_paths = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    image_paths.append(os.path.join(root, file))

        return image_paths

    def extract_labels_from_paths(self, all_paths, alcohol_keyword="alcohol"):
        y = [1 if alcohol_keyword in path else 0 for path in all_paths]
        return y

    def in_distribution_splits(self, train_image_paths, test_image_paths):
        all_paths = self.load_image_paths()
        self.train_paths = self.load_image_paths(train_image_paths)
        self.text_paths = self.load_image_paths(test_image_paths)
        # print(f"Train paths: {self.train_paths[:4]}")
        return  self.train_paths, self.train_paths


    def print_class_balance(self, paths, name):
        alcohol_count = sum(1 for path in paths if "alcohol" in path)
        non_alcohol = len(paths) - alcohol_count
        print(f"{name}: {non_alcohol / len(paths):.2%} non-alc, {alcohol_count / len(paths):.2%} alc")

    # def out_of_distribution_split(self):
    #     all_paths = self.load_image_paths()
    #     y = self.extract_labels_from_paths(all_paths, "alcohol")
    #     ood_train_paths, self.ood_test_paths, y_ood_train, y_ood_test = train_test_split(
    #         all_paths, y,
    #         test_size=self.test_size,
    #         stratify=y,  # Maintain class balance
    #         random_state=self.seed
    #     )
    #     self.ood_train_paths, self.ood_val_paths, y_ood_train, y_ood_val = train_test_split(
    #         ood_train_paths, y_ood_train,
    #         test_size=self.val_size,
    #         stratify=y_ood_train,  # Maintain class balance
    #         random_state=self.seed
    #     )
    #     return self.ood_train_paths, self.ood_val_paths, self.ood_test_paths
