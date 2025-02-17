# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:14:32 2025

@author: Usuario
"""
#First we will import all the libraries we will need
import pandas as pd
import nltk
from collections import Counter
import numpy as np
from scipy.stats import pearsonr, norm
nltk.download('punkt')

#Here we will load a CSV file, remove rows where the second column (index 1) contains 'streamelements' 
#(case-insensitive), drop the first three columns (keeping only the message column), tokenize all messages, 
#and returns a list of tokens. During tokenization, we also will remove punctuation so that only words (alphabetic tokens) 
#are retained. If a 'message' column exists it is used; otherwise, the fourth column (index 3) will be assumed to be 
#the message.
def load_and_tokenize(csv_file):

    try:
        df = pd.read_csv(csv_file, on_bad_lines='skip')
    except TypeError:
        df = pd.read_csv(csv_file, error_bad_lines=False)
    
    df = df[df.iloc[:, 1].astype(str).str.lower() != 'streamelements']
    
    if 'message' in df.columns:
        messages = df['message'].dropna().astype(str)
    else:
        messages = df.iloc[:, 3].dropna().astype(str)
    
    tokens = []
    for message in messages:
        raw_tokens = nltk.word_tokenize(message)
        word_tokens = [token for token in raw_tokens if token.isalpha()]
        tokens.extend(word_tokens)
    
    return tokens

# Here we will calculate the frequency and character length for each unique token.
# Then we will return a pandas DataFrame with columns: 'Token', 'Frequency', and 'Length'.

def get_token_statistics(tokens):

    freq_dict = Counter(tokens)
    data = [(token, freq, len(token)) for token, freq in freq_dict.items()]
    stats_df = pd.DataFrame(data, columns=['Token', 'Frequency', 'Length'])
    return stats_df

#Now we will Perform a Pearson correlation analysis between token frequency and token length.
#And return the correlation coefficient and p-value.


def perform_correlation_analysis(stats_df):

    correlation, p_value = pearsonr(stats_df['Frequency'], stats_df['Length'])
    return correlation, p_value


#Now we will compare two independent Pearson correlation coefficients using Fisher's r-to-z transformation.
#Parameters:
      #r1: correlation coefficient for group 1 (e.g., English)
      #n1: sample size (number of token types) for group 1
      #r2: correlation coefficient for group 2 (e.g., Spanish)
      #n2: sample size (number of token types) for group 2
      
#It will return:
      #diff: Difference (r1 - r2)
      #z_stat: The z-test statistic comparing the two correlations
      #p_value: Two-tailed p-value for the test

def compare_correlations(r1, n1, r2, n2):

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
#This is our main function
def main():
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



