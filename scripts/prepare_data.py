# scripts/prepare_data.py

import pandas as pd
import unicodedata
import re
from datasets import load_dataset
import os
from tqdm import tqdm

# Set pandas to show progress bars with tqdm
tqdm.pandas()

def clean_text(text):
    """
    Applies a series of cleaning steps to the input text,
    based on the pipeline from the project presentation.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Normalize Unicode (NFC) [cite: 52]
    text = unicodedata.normalize('NFC', text)
    
    # 2. Normalize punctuation and whitespace [cite: 56]
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 3. Remove control characters [cite: 52]
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    
    return text

def process_translation_data():
    """
    Downloads, cleans, and saves the English-Hindi translation dataset.
    """
    print("--- Starting Translation Data Processing ---")
    
    # Load the dataset from Hugging Face
    print("Downloading IIT-B English-Hindi dataset...")
    dataset = load_dataset("cfilt/iitb-english-hindi", split="train")
    df = dataset.to_pandas()
    
    initial_count = len(df)
    print(f"Initial number of pairs: {initial_count}")
    
    # Rename columns for clarity
    df = df.rename(columns={'translation': 'text'})
    df['english'] = df['text'].apply(lambda x: x['en'])
    df['hindi'] = df['text'].apply(lambda x: x['hi'])
    df = df[['english', 'hindi']]

    # Clean the text columns
    print("Cleaning text data...")
    df['english'] = df['english'].progress_apply(clean_text)
    df['hindi'] = df['hindi'].progress_apply(clean_text)

    # 3. Deduplicate sentence pairs [cite: 53]
    df.drop_duplicates(inplace=True)
    print(f"Pairs after deduplication: {len(df)}")
    
    # 4. Apply length filters [cite: 54]
    df['en_len'] = df['english'].str.split().str.len()
    df['hi_len'] = df['hindi'].str.split().str.len()
    
    df = df[(df['en_len'] > 1) & (df['en_len'] < 150)]
    df = df[(df['hi_len'] > 1) & (df['hi_len'] < 150)]
    
    # Remove pairs with extreme length ratios (e.g., >= 3:1) [cite: 54]
    length_ratio = df['en_len'] / df['hi_len']
    df = df[(length_ratio > 0.33) & (length_ratio < 3.0)]
    print(f"Pairs after length filtering: {len(df)}")

    # Select final columns and save
    final_df = df[['english', 'hindi']]
    output_path = os.path.join('data', 'translation_clean.csv')
    final_df.to_csv(output_path, index=False)
    
    print(f"✅ Clean translation data saved to {output_path}")
    print(f"Final pair count: {len(final_df)}")
    print("-" * 40)


def process_transliteration_data():
    """
    Cleans and saves the transliteration dataset from a local file.
    """
    print("\n--- Starting Transliteration Data Processing from Local File ---")
    
    # Define the path to your local file
    local_file_path = os.path.join('data', 'raw', 'hi.translit.sampled.train.tsv')
    
    print(f"Reading Dakshina dataset from {local_file_path}...")
    try:
        # Read the local tab-separated file (tsv)
        df = pd.read_csv(local_file_path, sep='\t', header=None, on_bad_lines='skip')
        df.columns = ['devanagari', 'roman', 'count']
        
    except FileNotFoundError:
        print(f"ERROR: The file was not found at {local_file_path}")
        print("Please make sure you have downloaded the data and placed it in the 'data/raw' folder.")
        return
    except Exception as e:
        print(f"Error processing the local file: {e}")
        return

    initial_count = len(df)
    print(f"Initial number of pairs: {initial_count}")
    
    # Select and clean relevant columns
    df = df[['roman', 'devanagari']]
    print("Cleaning text data...")
    df['roman'] = df['roman'].progress_apply(clean_text)
    df['devanagari'] = df['devanagari'].progress_apply(clean_text)

    # Deduplicate sentence pairs
    df.drop_duplicates(inplace=True)
    print(f"Pairs after deduplication: {len(df)}")
    
    # Drop empty rows that might result from cleaning
    df.dropna(inplace=True)
    df = df[df['roman'] != '']
    df = df[df['devanagari'] != '']
    
    output_path = os.path.join('data', 'transliteration_clean.csv')
    df.to_csv(output_path, index=False)
    
    print(f"✅ Clean transliteration data saved to {output_path}")
    print(f"Final pair count: {len(df)}")
    print("-" * 40)

if __name__ == "__main__":
    # Create the data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    process_translation_data()
    process_transliteration_data()
    
    print("\nData processing complete!")