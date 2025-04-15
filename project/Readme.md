# 🍷 Alcohol Content Detection in Images

# 🍷 Alcohol Detection from Images

Detect whether text in images refers to alcohol using OCR, correction, and a SetFit classifier.

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/your-username/alcohol-text-detector.git
cd alcohol-text-detector
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Folder Structure
Place your test images here:
```
images/
└── my_experiment/
    ├── alcohol/
    │   └── img001.jpg
    └── non_alcohol/
        └── img002.jpg
```

### 3. Configure & Run Test
In project/main.py, update the following:
```python
os.chdir(os.path.dirname(file_path))
input_csv = "project/results/all_spell_corrected_results.csv"
test_images_path = "images/"
output_csv_path = "results/results.csv"
model_save_path = "results/saved_model"
PipelineRunner.run_all_experiments(
    test_images_path=test_images_path,
    should_re_train=False
)
```

Then run:
```bash
python project/main.py
```

###  Output
Predictions are saved to:
```
results/results.csv
```

Example output:
```
file_name,prediction
img001.jpg,alcohol
img002.jpg,non_alcohol
```

## 1. About the project
The idea was to build something simple but effective, that can:
1. Read text from the image using OCR
2. Clean up OCR mistakes
3. Decide if the text is alcohol-related or not

---

## 2. A Bit About the Data

### What I worked with:

- All images had **overlayed text**
- Some were intentionally or unintentionally misspelled (e.g., `"lag3r"` instead of `"lager"`)
- The dataset was **balanced** (50% alcohol, 50% non-alcohol), but that's clearly not how things look in real life

---

### Real-World Context: Why It Matters

There are two main deployment scenarios I had in mind:

1. **Monitoring websites** for alcohol-related text → in most cases, the text won’t mention alcohol at all  
2. **Checking sites that claim to be alcohol-free** → assuming most people are honest, alcohol mentions would be rare, which again creates imbalance

So the model needs to take both **precision** and **recall** into account:
- Precision helps reduce the noise (false positives)
- Recall ensures we don’t miss subtle or obfuscated alcohol mentions

---

## 3. How I Broke It Down

Here’s the structure I followed:

### Step 1: Load the Images
- Nothing fancy — just basic loading and splitting into train/val/test
- I kept the class balance in the training data  
  (F1 already balances precision and recall, so that's where I focused)

### Step 2: OCR (Extracting Text)
- I used **PaddleOCR**, after comparing it with EasyOCR
- PaddleOCR gave better results, especially with bounding boxes and rotated text
- Also runs fine on CPU, which helped during development

Some useful OCR tweaks:
- `det_db_thresh = 0.2` → better for low-contrast text
- `det_db_box_thresh = 0.4` → filters noisy detections
- `det_db_unclip_ratio = 2.0` → useful for tightly spaced text

### Step 3: Fixing the Text

#### Character-level corrections:
- Cleaned up visual mistakes like `"0"` → `"O"`, `"1"` → `"I"`

#### Word-level corrections:
- Used **SymSpell**, a fast spell-checker
- I added a custom alcohol-related dictionary to bias corrections toward relevant words like `"vodc4"` → `"vodka"`

---

### 4. Classifying the Text

#### Try #1: Regex Rules

**Why I started here:**
Before jumping into neural networks and overly complex solutions, I wanted to see how far I could get with a simpler, more interpretable solution. Regex rules are lightweight, transparent, and easy to iterate on — which made them a solid starting point for this task.

In this case, a **handcrafted approach seemed reasonable**. The domain of alcohol references, while broad, is actually quite **well-defined**: we’re mostly looking for drink names, brands, or common phrases. It didn’t require deep semantic understanding, and I felt like I could cover most of the signal with clear, specific patterns.

I generated initial keyword lists using ChatGPT, then reviewed and edited them:
- Removed very short terms that could cause false positives
- Prioritized phrases that clearly indicate **non-alcohol content** before looking for alcohol-related ones
- Ignored whitespaces entirely when matching — since OCR often omits them, I treated the text as one long string and matched accordingly

**Result:**
- ✅ High precision — when it flagged something as alcohol, it was usually correct
- ❌ Low recall — missed many alcohol-related cases, likely due to aggressive filtering


#### Try #2: SetFit
While the regex-based solution was more computationally efficient — a clear advantage for large-scale data — this approach wasn’t far behind. It uses a small model and efficient fine-tuning, worked right out of the box and delivers good results and in hindsight, it probably should have been my starting point.

SetFit is known to work **exceptionally well with small datasets**, so I gave it a shot.
- It was very easy to  use it and I think I should have started with this solution as it was working "out of the box".
- Based on `paraphrase-MiniLM-L6-v2` for sentence embeddings (sentence transformers)
- Fine-tuned using contrastive learning + classification head
- Runs on CPU, trains fast, and doesn't require a huge labeled dataset

I used it **after text correction**, since I assumed the corrected text would tokenize more cleanly — and in practice, that did improve performance.

**Why I liked it:**
- Efficient to fine-tune — very few parameters are adjusted
- Model is small and quick to iterate with
- Captures **meaning**, not just keywords and generalizes across unseeen data (eg languages, brands, etc)

---

## 5. How I Evaluated It

- **F1 Score** was the main metric — balances precision and recall
- **Precision** mattered to avoid false alarms
- **Recall** helped identify subtle or hidden alcohol references
- I also observed the confusion matrix

---

## 6. Sample Results (Filling these in soon)

Method | F1 Score | Accuracy | Precision | Recall
Regex only (no speller/visual) | 0.571 | 0.721 | 0.884 | 0.422
Regex + Visual Fix | 0.690 | 0.784 | 0.942 | 0.544
Regex + Spell Correction | 0.690 | 0.784 | 0.942 | 0.544
Spell Correction + SetFit | [TBD] | [TBD] | [TBD] | [TBD]

**Takeaways**:
- My attempts with PaddleOCR outperformed EasyOCR in both quality and usability
- the regex based approaches arn't perfect but they tended to have good precision and very low recall meaning lot's of false negative. the model is cautios.
- The visual error correction helped in terms of accuracy precision and recall
- The spell correction didn't yeild any diffenrces
  - it was used afterwards in the SetFit model
- SetFit gave the most balanced results

---

## 7. What Was Still Hard

- Stylized fonts, curved layouts, and image noise were tough for OCR
- Merged words made it harder to match phrases
- Some alcohol words were intentionally obfuscated (hard to catch reliably)

---

## 8. What I’d Improve Next

- Refine the regex classification rules — some patterns may have been too aggressive
- Explore preprocessing steps that could simplify the OCR
