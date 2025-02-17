# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:14:32 2025

@author: Usuario
"""
import pandas as pd
import nltk
from collections import Counter
import numpy as np
from scipy.stats import pearsonr, norm

# Ensure the necessary NLTK tokenizer data is available.
nltk.download('punkt')

def load_and_tokenize(csv_file):
    """
    Loads a CSV file, removes rows where the second column (index 1)
    contains 'streamelements' (case-insensitive), drops the first three columns 
    (keeping only the message column), tokenizes all messages, and returns a list of tokens.
    
    During tokenization, punctuation is removed so that only words (alphabetic tokens) are retained.
    If a 'message' column exists it is used; otherwise, the fourth column (index 3) is assumed to be the message.
    """
    try:
        df = pd.read_csv(csv_file, on_bad_lines='skip')
    except TypeError:
        df = pd.read_csv(csv_file, error_bad_lines=False)
    
    # Filter out rows where the second column contains 'streamelements' (case-insensitive)
    df = df[df.iloc[:, 1].astype(str).str.lower() != 'streamelements']
    
    # Select only the message column: use 'message' if available, otherwise the fourth column.
    if 'message' in df.columns:
        messages = df['message'].dropna().astype(str)
    else:
        messages = df.iloc[:, 3].dropna().astype(str)
    
    tokens = []
    for message in messages:
        # Tokenize the message text.
        raw_tokens = nltk.word_tokenize(message)
        # Keep only alphabetic tokens (removes punctuation and tokens with numbers/symbols)
        word_tokens = [token for token in raw_tokens if token.isalpha()]
        tokens.extend(word_tokens)
    
    return tokens

def get_token_statistics(tokens):
    """
    Calculates the frequency and character length for each unique token.
    Returns a pandas DataFrame with columns: 'Token', 'Frequency', and 'Length'.
    """
    freq_dict = Counter(tokens)
    data = [(token, freq, len(token)) for token, freq in freq_dict.items()]
    stats_df = pd.DataFrame(data, columns=['Token', 'Frequency', 'Length'])
    return stats_df

def perform_correlation_analysis(stats_df):
    """
    Performs a Pearson correlation analysis between token frequency and token length.
    Returns the correlation coefficient and p-value.
    """
    correlation, p_value = pearsonr(stats_df['Frequency'], stats_df['Length'])
    return correlation, p_value

def compare_correlations(r1, n1, r2, n2):
    """
    Compares two independent Pearson correlation coefficients using Fisher's r-to-z transformation.
    
    Parameters:
      r1: correlation coefficient for group 1 (e.g., English)
      n1: sample size (number of token types) for group 1
      r2: correlation coefficient for group 2 (e.g., Spanish)
      n2: sample size (number of token types) for group 2
      
    Returns:
      diff: Difference (r1 - r2)
      z_stat: The z-test statistic comparing the two correlations
      p_value: Two-tailed p-value for the test
    """
    # Fisher r-to-z transformation:
    def fisher_z(r):
        return 0.5 * np.log((1 + r) / (1 - r))
    
    z1 = fisher_z(r1)
    z2 = fisher_z(r2)
    # Standard errors for the z-values
    se1 = 1 / np.sqrt(n1 - 3)
    se2 = 1 / np.sqrt(n2 - 3)
    se_diff = np.sqrt(se1**2 + se2**2)
    z_stat = (z1 - z2) / se_diff
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    return r1 - r2, z_stat, p_value

def main():
    # Define file paths for the English and Spanish CSVs.
    english_file = 'english.csv'
    spanish_file = 'spanish.csv'
    
    # --- Process English Data ---
    print("Processing English data...\n")
    english_tokens = load_and_tokenize(english_file)
    english_stats = get_token_statistics(english_tokens)
    english_corr, english_p = perform_correlation_analysis(english_stats)
    english_n = len(english_stats)  # Number of token types in English data
    
    print(f"English Pearson correlation between frequency and token length: {english_corr:.3f} (p = {english_p:.3e})")
    # Top 10 most frequent tokens in English
    top10_english = english_stats.sort_values(by='Frequency', ascending=False).head(10)
    print("\nTop 10 most frequent tokens in English:")
    print(top10_english.to_string(index=False))
    print("\n" + "="*60 + "\n")
    
    # --- Process Spanish Data ---
    print("Processing Spanish data...\n")
    spanish_tokens = load_and_tokenize(spanish_file)
    spanish_stats = get_token_statistics(spanish_tokens)
    spanish_corr, spanish_p = perform_correlation_analysis(spanish_stats)
    spanish_n = len(spanish_stats)  # Number of token types in Spanish data
    
    print(f"Spanish Pearson correlation between frequency and token length: {spanish_corr:.3f} (p = {spanish_p:.3e})")
    # Top 10 most frequent tokens in Spanish
    top10_spanish = spanish_stats.sort_values(by='Frequency', ascending=False).head(10)
    print("\nTop 10 most frequent tokens in Spanish:")
    print(top10_spanish.to_string(index=False))
    print("\n" + "="*60 + "\n")
    
    # --- Compare the Two Correlations ---
    diff, z_stat, p_value = compare_correlations(english_corr, english_n, spanish_corr, spanish_n)
    print("Comparison between English and Spanish Zipf’s Law of Abbreviation:")
    print(f"Difference in correlation coefficients (English - Spanish): {diff:.3f}")
    print(f"Z-statistic: {z_stat:.3f}")
    print(f"P-value for difference: {p_value:.3e}")

if __name__ == '__main__':
    main()



