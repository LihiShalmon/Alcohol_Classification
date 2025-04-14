
# pipeline/classifier.py
import re
from config import regex_classification_terms, non_alcohol_exclude


class TextClassifier:
    def __init__(self, alcohol_keywords=regex_classification_terms,
                 classification_type="regex", advanced_spell_correction = False):
        
        self.alcohol_keywords = alcohol_keywords

    def generate_all_regex_predictions(self, images_captions):
        all_results = []
        for cur_caption in images_captions:
            cur_image_matches = self.search_for_image_matches(cur_caption)
            match_strings = [str(m).lower() for m in cur_image_matches]

            is_non_alcohol = any("non_alcohol" in s for s in match_strings)

            prediction = "non_alcohol" if is_non_alcohol else "alcohol"

            all_results.append(self.return_classification_results(cur_image_matches, prediction))
            
        return all_results

    def search_for_image_matches(self, text):
        cur_image_matches = []
        for keyword in non_alcohol_exclude["non_alcohol_exclude"]:
            pattern = rf"{re.escape(keyword)}"
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                cur_image_matches.append({
                    "word": f"{match.group(0)}_NON_ALCOHOL",
                    "category": "non_alcohol_exclude"
                })

        for category_name, keywords in self.alcohol_keywords.items():
            for keyword in keywords:
                pattern = rf"{keyword}"
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    cur_image_matches.append({
                        "word": match.group(0),
                        "category": category_name
                    })

        if not cur_image_matches:
            cur_image_matches.append({
                "word": "NO_MATCH",
                "category": "NO_MATCH_NON_ALCOHOL"
            })

        return cur_image_matches


    def return_classification_results(self, cur_image_matches, prediction):
        result = {
                "prediction": prediction,
                "matches": cur_image_matches,
                "matched_words": ",".join(m["word"] for m in cur_image_matches) if cur_image_matches else "",
                "matched_categories": ",".join(set(m["category"] for m in cur_image_matches)) if cur_image_matches else ""
            }
        return result
            
        