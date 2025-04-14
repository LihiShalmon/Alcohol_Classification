import json
import csv
import re
from pathlib import Path

def clean_string(s):
    # Remove escaped characters and extra quotes
    s = re.sub(r'\\[nt"]', ' ', s)
    s = s.replace('\\"', '"').replace("\\'", "'")
    s = s.replace('\\\\', '\\')
    return s.strip()

def flatten_dict(d, parent_key='', sep='_'):
    # Flattens nested dictionaries
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def json_to_clean_csv(json_path, csv_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = f.read()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("Trying to fix malformed JSON...")
        raw = clean_string(raw)
        data = json.loads(raw)

    # If it's a wandb.Table, data might be {'data': [...], 'columns': [...]}
    if isinstance(data, dict) and 'data' in data and 'columns' in data:
        records = [dict(zip(data['columns'], row)) for row in data['data']]
    elif isinstance(data, list):
        records = data
    else:
        records = [data]

    cleaned_records = []
    for r in records:
        flat = flatten_dict(r)
        cleaned = {k: clean_string(str(v)) if isinstance(v, str) else v for k, v in flat.items()}
        cleaned_records.append(cleaned)

    keys = sorted(set().union(*(r.keys() for r in cleaned_records)))

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(cleaned_records)


json_to_clean_csv(r'project\results\all_spell_corrected_results.json',
             r'project\results\all_spell_corrected_results.csv')
