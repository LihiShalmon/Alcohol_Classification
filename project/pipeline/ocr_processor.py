# pipeline/ocr_processor.py
import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
# text_corrector.py
import re
from typing import Dict, List, Optional
from symspellpy import SymSpell, Verbosity
import pkg_resources
import config
matplotlib.use('Agg')  # Set non-interactive backend

# Define a type for text corrector to use in type hints
TextCorrectorType = Any  # This could be refined if needed

def draw_ocr(pil_image, boxes, texts, scores, font_path=None):
    """
    Draws OCR results on the image
    Returns: PIL Image with annotations
    """
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


class OCRProcessor:
    """Handles image OCR processing."""
    
    def __init__(
        self,
        ocr_config: Dict[str, Any],
        text_corrector: Optional[TextCorrectorType] = None,
        font_path: str = "arial.ttf"
    ):
        """
        Initialize OCR processor.

        """
        self.ocr = PaddleOCR(**ocr_config)
        self.text_corrector = text_corrector
        self.font_path = font_path if font_path and os.path.exists(font_path) else None
    
    def get_variants(self, image_bgr):
        """Generate image variants for OCR processing."""
        # gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return {
            'original': image_bgr,
            # 'gray': gray,
        }
    
    def process_single_variant(self, image_bgr: np.ndarray, title: str) -> Tuple[Image.Image, str]:
        """
        Process a single image variant with OCR.
        
        Args:
            image_bgr: BGR image array
            title: Title of the variant
        
        Returns:
            Tuple of (annotated image, extracted text)
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        try:
            result = self.ocr.ocr(image_rgb, cls=True)
            
            # Handle all potential empty result cases
            if not result or len(result) == 0 or result[0] is None or len(result[0]) == 0:
                # Return empty text for no detections
                return Image.fromarray(image_rgb), ""
            
            lines = result[0]
            valid_lines = []
            for line in lines:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    # Make sure the line contains both box coordinates and text+score
                    if isinstance(line[0], (list, tuple)) and isinstance(line[1],
                                                                          (list, tuple)) and len(line[1]) >= 2:
                        valid_lines.append(line)
            
            # If no valid lines found after filtering
            if not valid_lines:
                return Image.fromarray(image_rgb), ""
            
            # Process valid lines
            boxes = [line[0] for line in valid_lines]
            texts = [line[1][0] for line in valid_lines]
            scores = [line[1][1] for line in valid_lines]
            
            # Apply text correction if available
            if self.text_corrector and hasattr(self.text_corrector, 'correct_text'):
                texts = [self.text_corrector.correct_text(text) for text in texts]
            
            combined_text = ", ".join(texts).lower()
            
            # Draw annotations on the image
            box_annotated_instance = draw_ocr(
                Image.fromarray(image_rgb), boxes, texts, scores, font_path=self.font_path
            )
            
            return box_annotated_instance, combined_text
            
        except Exception as e:
            print(f"Error in OCR processing for {title}: {e}")
            # Return original image with empty text on any exception
            return Image.fromarray(image_rgb), ""
    
    def process_image(self, img_path: str, save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process image with OCR.
        
        Args:
            img_path: Path to image file
            save_dir: Optional directory to save visualization
        
        Returns:
            Dictionary with OCR results
        """
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
        
        # Save visualization if directory provided
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
    


class TextCorrect:
    """Handles text correction for OCR output."""
    
    def __init__(
        self,
        use_basic_correction: bool = True,
        use_advanced_correction: bool = False,
        custom_substitutions: Optional[Dict[str, str]] = None,
        dictionary_terms: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize text corrector.
        
        Args:
            use_basic_correction: Whether to use basic character substitution
            use_advanced_correction: Whether to use advanced spell checking
            custom_substitutions: Custom character substitutions dictionary
            dictionary_terms: Domain-specific terms to add to the spell checker
        """
        self.use_basic_correction = use_basic_correction
        self.use_advanced_correction = use_advanced_correction
        
        # Default OCR mistake corrections
        self.substitutions = config.ocr_config
        
        # Initialize sym_spell if enabled
        if use_advanced_correction:
            self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=3)
            dictionary_path = pkg_resources.resource_filename(
                "symspellpy", "frequency_dictionary_en_82_765.txt"
            )
            self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
            
            # Add domain-specific terms to dictionary
            if dictionary_terms:
                for category, terms in dictionary_terms.items():
                    for term in terms:
                        self.sym_spell.create_dictionary_entry(term, 10000)
    
    def correct_text(self, text: str) -> str:
        """
        Apply text correction to OCR output.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Corrected text
        """
        if not text:
            return text
        
        # Apply basic character substitution
        if self.use_basic_correction:
            text = self._apply_basic_correction(text)
        
        # Apply advanced spell checking
        if self.use_advanced_correction:
            text = self._apply_advanced_correction(text)
        
        return text
    
    def _apply_basic_correction(self, text: str) -> str:
        """Apply basic character substitutions."""
        for wrong, right in sorted(
            self.substitutions.items(), key=lambda x: len(x[0]), reverse=True
        ):
            text = re.sub(re.escape(wrong), right, text)
        return text
    
    def _apply_advanced_correction(self, text: str) -> str:
        """Apply advanced spell checking."""
        words = text.split()
        corrected_words = []
        
        for word in words:
            segmented = self.sym_spell.word_segmentation(word) #inserting missing spaces

            compound = self.sym_spell.lookup_compound( # correction of multi-word input
                segmented.corrected_string, max_edit_distance=2
            )
            
            # Get suggestions for the term
            suggestions = self.sym_spell.lookup(
                compound[0].term, Verbosity.CLOSEST, max_edit_distance=2
            )
            
            if suggestions:
                # Use the top suggestion
                top_suggestions = [s.term for s in suggestions[:3]]
                corrected_words.append(top_suggestions[0])
            else:
                # Keep original word if no suggestions
                corrected_words.append(word)
        
        return " ".join(corrected_words)
# import os
# import cv2
# import numpy as np
# from paddleocr import PaddleOCR
# from PIL import ImageDraw, ImageFont, Image
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')  # Set non-interactive backend
# import re
# from symspellpy import SymSpell, Verbosity
# from config import regex_classification_terms
# import pkg_resources
# from config import ocr_config


# class OCRProcessor:
#     def __init__(self, predefined_config, font_path="arial.ttf", symspell_correction=False):
#         self.ocr = PaddleOCR(**predefined_config)
#         self.font_path = font_path if font_path and os.path.exists(font_path) else None
#         self.advanced_spell_correction = symspell_correction
#         if symspell_correction:
#             self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=3)
#             dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
#             self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
#             for category, terms in regex_classification_terms.items():
#                 for term in terms:
#                     self.sym_spell.create_dictionary_entry(term, 10000)

#     def correct_text(self, text):
#         for wrong, right in sorted(ocr_config["OCR_CHARACTER_CORRECTIONS"].items(), key=lambda x: len(x[0]), reverse=True):
#             text = re.sub(re.escape(wrong), right, text)
#         if self.advanced_spell_correction:
#             text = self.added_spell_correction(text)
#         return text

#     def added_spell_correction(self, text):
#         words = text.split()
#         corrected_words = []
#         for word in words:
#             segmented = self.sym_spell.word_segmentation(word)
#             compound = self.sym_spell.lookup_compound(segmented.corrected_string, max_edit_distance=2)
#             suggestions = self.sym_spell.lookup(compound[0].term, Verbosity.CLOSEST, max_edit_distance=2)
#             if suggestions:
#                 top_suggestions = [s.term for s in suggestions[:3]]  # Top 3 suggestions
#                 corrected_words.append(top_suggestions[0])  # ← you can choose top-1, or try logic to pick best
#             else:
#                 corrected_words.append(word)
#         return " ".join(corrected_words)

#     def get_variants(self, image_bgr):
#         gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#         return {
#             'original': image_bgr,
#             'gray': gray,
#         }
    
#     def process_single_variant(self, image_bgr, title):
#         # perform ocr
#         image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
#         try:
#             result = self.ocr.ocr(image_rgb, cls=True)
            
#             # Handle all potential empty result cases
#             if not result or len(result) == 0 or result[0] is None or len(result[0]) == 0:
#                 # Return empty text for no detections
#                 return Image.fromarray(image_rgb), ""
            
#             # Extract OCR results
#             lines = result[0]
            
#             # Validate each line has the expected structure
#             valid_lines = []
#             for line in lines:
#                 if isinstance(line, (list, tuple)) and len(line) >= 2:
#                     # Make sure the line contains both box coordinates and text+score
#                     if isinstance(line[0], (list, tuple)) and isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
#                         valid_lines.append(line)
            
#             # If no valid lines found after filtering
#             if not valid_lines:
#                 return Image.fromarray(image_rgb), ""
            
#             # Process valid lines
#             boxes = [line[0] for line in valid_lines]
#             texts = [self.correct_text(line[1][0]) for line in valid_lines]
#             scores = [line[1][1] for line in valid_lines]
#             combined_text = ", ".join(texts).lower()
            
#             # Draw annotations on the image
#             box_annotated_instance = draw_ocr(Image.fromarray(image_rgb), boxes, texts, scores,
#                                 font_path=self.font_path)
            
#             return box_annotated_instance, combined_text
            
#         except Exception as e:
#             print(f"Error in OCR processing for {title}: {e}")
#             # Return original image with empty text on any exception
#             return Image.fromarray(image_rgb), ""
        
        
#     def process_image(self, img_path, save_dir=None):
#         if not os.path.exists(img_path):
#             raise FileNotFoundError(img_path)

#         image_bgr = cv2.imread(img_path)
#         variants = self.get_variants(image_bgr)

#         all_texts = []
#         fig = plt.figure(figsize=(15, 10))
#         for i, (name, variant_img) in enumerate(variants.items(), 1):
#             annotated, text = self.process_single_variant(variant_img, name)
#             all_texts.append(text)
#             ax = fig.add_subplot(2, 3, i)
#             ax.imshow(annotated)
#             ax.set_title(name)
#             ax.axis('off')

#         combined_text = " ".join(all_texts)
#         label = os.path.basename(os.path.dirname(img_path))
#         print(f"Label: {label}, OCR Text: {img_path}")
#         if save_dir:
#             os.makedirs(save_dir, exist_ok=True)
#             filename = os.path.basename(img_path).replace('.jpg', '_ocr.png')
#             vis_path = os.path.join(save_dir, f"{label}_{filename}")
#             fig.tight_layout()
#             fig.savefig(vis_path)
#             plt.close(fig)
#         else:
#             vis_path = None
#             plt.show()

#         return {
#             'image_path': img_path,
#             'ocr_text': combined_text,
#             'ocr_visualization': vis_path
#         }

# def draw_ocr(pil_image, boxes, texts, scores, font_path=None):
#     draw = ImageDraw.Draw(pil_image)
#     font = ImageFont.truetype(font_path, 18) if font_path and os.path.exists(font_path) else ImageFont.load_default()

#     for box, text, score in zip(boxes, texts, scores):
#         x_coords = [p[0] for p in box]
#         y_coords = [p[1] for p in box]
#         min_x, max_x = min(x_coords), max(x_coords)
#         min_y, max_y = min(y_coords), max(y_coords)

#         draw.rectangle([(min_x, min_y), (max_x, max_y)], outline="red", width=2)
#         draw.text((min_x, min_y - 20), f"{text} ({score:.2f})", fill="red", font=font)

#     return pil_image
