import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import ImageDraw, ImageFont, Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import re
from symspellpy import SymSpell, Verbosity
from config import regex_classification_terms
import pkg_resources

COMMON_OCR_MISTAKES = {
    "0": "O", "1": "I", "2": "Z", "3": "E", "4": "A",
    "5": "S", "6": "G", "7": "T", "8": "B", "9": "g",
    "l": "I", "vv": "w",
}

class OCRProcessor:
    def __init__(self, predefined_config, font_path="arial.ttf", advanced_spell_correction=False):
        self.ocr = PaddleOCR(**predefined_config)
        self.font_path = font_path if font_path and os.path.exists(font_path) else None
        self.advanced_spell_correction = advanced_spell_correction
        if advanced_spell_correction:
            self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=3)
            dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
            self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
            for category, terms in regex_classification_terms.items():
                for term in terms:
                    self.sym_spell.create_dictionary_entry(term, 10000)

    def correct_text(self, text):
        for wrong, right in sorted(COMMON_OCR_MISTAKES.items(), key=lambda x: len(x[0]), reverse=True):
            text = re.sub(re.escape(wrong), right, text)
        if self.advanced_spell_correction:
            text = self.added_spell_correction(text)
        return text

    def added_spell_correction(self, text):
        words = text.split()
        corrected_words = []
        for word in words:
            segmented = self.sym_spell.word_segmentation(word)
            compound = self.sym_spell.lookup_compound(segmented.corrected_string, max_edit_distance=2)
            suggestions = self.sym_spell.lookup(compound[0].term, Verbosity.CLOSEST, max_edit_distance=2)
            if suggestions:
                top_suggestions = [s.term for s in suggestions[:3]]  # Top 3 suggestions
                corrected_words.append(top_suggestions[0])  # ← you can choose top-1, or try logic to pick best
            else:
                corrected_words.append(word)
        return " ".join(corrected_words)

    def get_variants(self, image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return {
            'original': image_bgr,
            'gray': gray,
        }
    
    def process_single_variant(self, image_bgr, title):
        # perform ocr
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        try:
            result = self.ocr.ocr(image_rgb, cls=True)
            
            # Handle all potential empty result cases
            if not result or len(result) == 0 or result[0] is None or len(result[0]) == 0:
                # Return empty text for no detections
                return Image.fromarray(image_rgb), ""
            
            # Extract OCR results
            lines = result[0]
            
            # Validate each line has the expected structure
            valid_lines = []
            for line in lines:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    # Make sure the line contains both box coordinates and text+score
                    if isinstance(line[0], (list, tuple)) and isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                        valid_lines.append(line)
            
            # If no valid lines found after filtering
            if not valid_lines:
                return Image.fromarray(image_rgb), ""
            
            # Process valid lines
            boxes = [line[0] for line in valid_lines]
            texts = [self.correct_text(line[1][0]) for line in valid_lines]
            scores = [line[1][1] for line in valid_lines]
            combined_text = ", ".join(texts).lower()
            
            # Draw annotations on the image
            box_annotated_instance = draw_ocr(Image.fromarray(image_rgb), boxes, texts, scores,
                                font_path=self.font_path)
            
            return box_annotated_instance, combined_text
            
        except Exception as e:
            print(f"Error in OCR processing for {title}: {e}")
            # Return original image with empty text on any exception
            return Image.fromarray(image_rgb), ""
        
        
    def process_image(self, img_path, save_dir=None):
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)

        image_bgr = cv2.imread(img_path)
        variants = self.get_variants(image_bgr)

        all_texts = []
        fig = plt.figure(figsize=(15, 10))
        for i, (name, variant_img) in enumerate(variants.items(), 1):
            annotated, text = self.process_single_variant(variant_img, name)
            all_texts.append(text)
            ax = fig.add_subplot(2, 3, i)
            ax.imshow(annotated)
            ax.set_title(name)
            ax.axis('off')

        combined_text = " ".join(all_texts)
        label = os.path.basename(os.path.dirname(img_path))
        print(f"Label: {label}, OCR Text: {img_path}")
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.basename(img_path).replace('.jpg', '_ocr.png')
            vis_path = os.path.join(save_dir, f"{label}_{filename}")
            fig.tight_layout()
            fig.savefig(vis_path)
            plt.close(fig)
        else:
            vis_path = None
            plt.show()

        return {
            'image_path': img_path,
            'ocr_text': combined_text,
            'ocr_visualization': vis_path
        }

def draw_ocr(pil_image, boxes, texts, scores, font_path=None):
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(font_path, 18) if font_path and os.path.exists(font_path) else ImageFont.load_default()

    for box, text, score in zip(boxes, texts, scores):
        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        draw.rectangle([(min_x, min_y), (max_x, max_y)], outline="red", width=2)
        draw.text((min_x, min_y - 20), f"{text} ({score:.2f})", fill="red", font=font)

    return pil_image

# import os
# import cv2
# import numpy as np
# from paddleocr import PaddleOCR, draw_ocr
# from PIL import ImageDraw, ImageFont
# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')  # Set the non-interactive backend globally
# import re
# from symspellpy import SymSpell, Verbosity
# from config import regex_classification_terms
# import pkg_resources

# COMMON_OCR_MISTAKES = {
#     "0": "O", "1": "I", "2": "Z", "3": "E", "4": "A",
#     "5": "S", "6": "G", "7": "T", "8": "B", "9": "g",
#     "l": "I", "vv": "w",
# }

# class OCRProcessor:
#     def __init__(self,predefined_config, font_path="arial.ttf",
#                   advanced_spell_correction= False):
        
#         self.ocr = PaddleOCR( **predefined_config)
#         self.font_path = font_path if font_path and os.path.exists(font_path) else None
#         self.advanced_spell_correction = advanced_spell_correction
#         if advanced_spell_correction:
#             self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
#             self.dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
#             self.sym_spell.load_dictionary(self.dictionary_path, term_index=0, count_index=1)
        
    
#     def correct_text(self, text):
#         "Replaes the OCR extracted text with a reasonable option"
#         "Based on charachter shape -eg 3 --> to E " 
#         sorted_replacements = sorted(COMMON_OCR_MISTAKES.items(),
#                                       key=lambda x: len(x[0]), reverse=True)
#         for wrong, right in sorted_replacements:
#             text = re.sub(re.escape(wrong), right, text)
#         if self.advanced_spell_correction:
#             text = self.added_spell_correction(text)

#         return text

#     def get_variants(self, image_bgr):
#         gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#         # contrast = cv2.convertScaleAbs(image_bgr, alpha=1.5, beta=10)
#         # equalized = cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR)
#         # hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
       
#         return {
#             'original': image_bgr,
#             'gray': gray,
#             # 'contrast': contrast,
#             # 'equalized': equalized,
#             # "hsv":hsv
#         }

#     def process_single_variant(self, image_bgr, title):
#         # perform ocr
#         # image_rgb = image_bgr
#         image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#         result = self.ocr.ocr(image_rgb, cls=True)
#         if not result or not result[0]:
#             return Image.fromarray(image_rgb), ""
        
#         # check results
#         lines = result[0]
#         boxes = [line[0] for line in lines]
#         texts = [self.correct_text(line[1][0]) for line in lines]
#         scores = [line[1][1] for line in lines]
#         combined_text = ", ".join(texts).lower()
#         # visualize 
#         box_annotated_instance = draw_ocr(Image.fromarray(image_rgb), boxes, texts, scores,
#                               font_path=self.font_path)
#         return box_annotated_instance, combined_text

#     def process_image(self, img_path, save_dir=None):
#         if not os.path.exists(img_path):
#             raise FileNotFoundError(img_path)

#         image_bgr = cv2.imread(img_path)
#         variants = self.get_variants(image_bgr)

#         all_texts = []
#         fig = plt.figure(figsize=(15, 10))
#         for i, (name, variant_img) in enumerate(variants.items(), 1):
#             # draw sigle varience instance
#             annotated, text = self.process_single_variant(variant_img, name)
#             all_texts.append(text)
#             ax = fig.add_subplot(2, 3, i)
#             ax.imshow(annotated)
#             ax.set_title(name)
#             ax.axis('off')

#         combined_text = " ".join(all_texts)
#         label = img_path.split("\\")[-2]
#         print(f"Label: {label}, OCR Text: {img_path}")
#         if save_dir:
#             os.makedirs(save_dir, exist_ok=True)
#             filename = os.path.basename(img_path).replace('.jpg', '_ocr.png')
#             vis_path = os.path.join(save_dir, f"{label}_{filename}")
#             fig.tight_layout()
#             fig.savefig(f"{vis_path}")
#             plt.close(fig)
#         else:
#             vis_path = None
#             plt.show()

#         return {
#             'image_path': img_path,
#             'ocr_text': combined_text,
#             'ocr_visualization': vis_path
#         }


#     def added_spell_correction(self,text):

#         # Add alcohol-related terms to the dictionary
#         for category, terms in regex_classification_terms.items():
#             for term in terms:
#                 self.sym_spell.create_dictionary_entry(term, 10000)
        
#         # Process each word
#         words = text.split()
#         corrected_words = []
#         for word in words:
#             segmented = self.sym_spell.word_segmentation(word)
#             corrected = self.sym_spell.lookup_compound(segmented.corrected_string, max_edit_distance=2)

#             suggestions = self.sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
#             if suggestions:
#                 corrected_words.append(suggestions[0].term)
#             else:
#                 corrected_words.append(word)
        
#         return " ".join(corrected_words)


# def draw_ocr(pil_image, boxes, texts, scores, font_path=None):
#     """
#     Minimal fallback for draw_ocr if paddleocr.tools.visualize.draw_ocr is unavailable.
#     Draw bounding rectangles and text onto a PIL image.
#     """
#     draw = ImageDraw.Draw(pil_image)

#     # Use a default font if none is specified or if font_path doesn't exist
#     if font_path and os.path.exists(font_path):
#         font = ImageFont.truetype(font_path, 18)
#     else:
#         font = ImageFont.load_default()

#     # Zip together all bounding boxes, recognized texts, and confidence scores
#     for box, text, score in zip(boxes, texts, scores):
#         # Each box is [[x1, y1],[x2, y2],[x3, y3],[x4, y4]]
#         # We'll just draw a rectangle around the box:
#         x_coords = [p[0] for p in box]
#         y_coords = [p[1] for p in box]
#         min_x, max_x = min(x_coords), max(x_coords)
#         min_y, max_y = min(y_coords), max(y_coords)

#         # Outline the bounding box in red
#         draw.rectangle(
#             [(min_x, min_y), (max_x, max_y)],
#             outline="red",
#             width=2
#         )

#         # Place text above the box
#         label = f"{text} ({score:.2f})"
#         draw.text((min_x, min_y - 20), label, fill="red", font=font)

#     return pil_image
