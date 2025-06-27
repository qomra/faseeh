#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Arabic Tokenizer Evaluation Script
==================================
This script evaluates the performance of different tokenizers on Arabic text datasets.
It measures metrics like: 
- Total token count
- Fertility score 
- Tashkeel preservation
- Processing speed
"""

import argparse
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# HuggingFace imports
import torch
from transformers import AutoTokenizer
from datasets import load_dataset


class TokenizerEvaluator:
    """Evaluates tokenizers on Arabic text datasets."""
    
    def __init__(
        self, 
        custom_tokenizer_path: Optional[str] = None, 
        datasets: Optional[List[str]] = None
    ):
        """
        Initialize the evaluator with tokenizers and datasets.
        
        Args:
            custom_tokenizer_path: Path to your custom tokenizer
            datasets: List of dataset names to use for evaluation
        """
        # Default reference tokenizers if none provided
        # self.reference_tokenizers = reference_tokenizers or [
        #     "FreedomIntelligence/AceGPT-v1.5-13B-Chat",
        #     "riotu-lab/Aranizer-SP-86k",
        #     "bert-base-multilingual-cased",
        #     "google/mt5-base"
        # ]
        
        # Add custom tokenizer if provided
        self.tokenizers = {}
        self.custom_tokenizer_path = custom_tokenizer_path
        
        # Default datasets if none provided
        self.datasets = datasets or [
          ("./datasets/MohamedRashad/rasaif-translations","arabic"),
            ("./datasets/HeshamHaroon/arabic-quotes","quote"),
           ("./datasets/SaiedAlshahrani/Moroccan_Arabic_Wikipedia_20230101_nobots","text")
        ]
        
        # Results storage
        self.results = {}
        
    def load_tokenizers(self):
        """Load all tokenizers for evaluation."""
        print("Loading tokenizers...")
        
        # Load custom tokenizer if provided
        if self.custom_tokenizer_path:
            try:
                self.tokenizers["Mysam/Oryx-2.0"] = AutoTokenizer.from_pretrained(self.custom_tokenizer_path)
                print(f"Successfully loaded custom tokenizer from {self.custom_tokenizer_path}")
            except Exception as e:
                print(f"Error loading custom tokenizer: {e}")
                
        print(f"Loaded {len(self.tokenizers)} tokenizers successfully.")
        
    def _count_words(self, text: str) -> int:
        """Count the number of words in Arabic text."""
        # Basic word counting for Arabic (space-separated)
        return len(text.split())
    
    def _has_tashkeel(self, original: str, tokenized: List[str]) -> bool:
        """Check if tashkeel is preserved after tokenization and detokenization."""
        # List of Arabic diacritics (tashkeel)
        tashkeel_chars = ['َ', 'ً', 'ُ', 'ٌ', 'ِ', 'ٍ', 'ْ', 'ّ']
        
        # Check if any tashkeel exists in original
        has_tashkeel_original = any(char in original for char in tashkeel_chars)
        
        if not has_tashkeel_original:
            # No tashkeel in original, so can't test preservation
            return None
        
        # Check if joined tokens contain any tashkeel
        joined = ''.join(tokenized)
        has_tashkeel_tokens = any(char in joined for char in tashkeel_chars)
        
        return has_tashkeel_tokens
    
    def evaluate_tokenizer(self, tokenizer_name: str, texts: List[str]) -> Dict[str, Any]:
        """
        Evaluate a single tokenizer on a list of texts.
        
        Args:
            tokenizer_name: Name of the tokenizer to evaluate
            texts: List of Arabic texts for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        tokenizer = self.tokenizers[tokenizer_name]
        total_tokens = 0
        total_words = 0
        processing_time = 0
        tashkeel_samples = 0
        tashkeel_preserved = 0
        
        # Process each text
        for text in texts:
            # Skip empty texts
            if not text or len(text.strip()) == 0:
                continue
                
            # Count words
            word_count = self._count_words(text)
            total_words += word_count
            
            # Measure tokenization time
            start_time = time.time()
            encoded = tokenizer.encode(text)
            processing_time += (time.time() - start_time)
            
            # Count tokens
            total_tokens += len(encoded)
            
            # Check tashkeel preservation (for a sample of texts)
            if np.random.random() < 0.1:  # Check 10% of texts for tashkeel
                decoded_tokens = tokenizer.convert_ids_to_tokens(encoded)
                tashkeel_result = self._has_tashkeel(text, decoded_tokens)
                if tashkeel_result is not None:
                    tashkeel_samples += 1
                    if tashkeel_result:
                        tashkeel_preserved += 1
        
        # Calculate metrics
        fertility_score = total_tokens / total_words if total_words > 0 else float('inf')
        print(f"total number of words: {total_words}")
        tashkeel_score = tashkeel_preserved / tashkeel_samples if tashkeel_samples > 0 else None
        tokens_per_second = total_tokens / processing_time if processing_time > 0 else 0
        
        # Get tokenizer class and vocabulary size
        tokenizer_class = tokenizer.__class__.__name__
        vocab_size = len(tokenizer.get_vocab()) if hasattr(tokenizer, 'get_vocab') else None
        
        return {
            "model_name": tokenizer_name,
            "tokenizer_class": tokenizer_class,
            "vocab_size": vocab_size,
            "total_tokens": total_tokens,
            "total_words": total_words,
            "fertility_score": fertility_score,
            "preserves_tashkeel": "✅" if tashkeel_score and tashkeel_score > 0.5 else "❌",
            "tashkeel_preservation_rate": tashkeel_score,
            "tokens_per_second": tokens_per_second
        }
    
    def load_dataset_texts(self, dataset_name: str) -> List[str]:
        """
        Load texts from a dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            List of texts from the dataset
        """
        try:
            ds = load_dataset(dataset_name)
            return ds
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return []
        
    def run_evaluation(self):
        """Run the full evaluation process."""
        self.load_tokenizers()
        results = []
        all_texts = []
        for dataset_name,dataset_col in self.datasets:
            print(f"Evaluating on dataset: {dataset_name}")
            texts = self.load_dataset_texts(dataset_name)["train"][dataset_col]
            all_texts += texts
            print(f"Loaded {len(texts)} texts from {dataset_name}")
            
        # Evaluate each tokenizer
        for tokenizer_name in self.tokenizers:
            print(f"Evaluating tokenizer: {tokenizer_name}")
            result = self.evaluate_tokenizer(tokenizer_name, all_texts)
            result["dataset"] = dataset_name
            results.append(result)
                
        # Create dataframe with all results
        self.results_df = pd.DataFrame(results)
        return self.results_df
    
    def save_results(self, output_path: str = "tokenizer_evaluation_results.csv"):
        """Save results to CSV file."""
        if hasattr(self, 'results_df'):
            self.results_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        else:
            print("No results to save. Run evaluation first.")
    
    def print_leaderboard(self):
        """Print a formatted leaderboard of tokenizer performance."""
        if not hasattr(self, 'results_df'):
            print("No results available. Run evaluation first.")
            return
            
        # Group by tokenizer and aggregate across datasets
        grouped = self.results_df.groupby('model_name').agg({
            'tokenizer_class': 'first',
            'vocab_size': 'first',
            'total_tokens': 'sum',
            'total_words': 'sum',
            'fertility_score': 'mean',
            'preserves_tashkeel': 'first',
            'tokens_per_second': 'mean'
        }).reset_index()
        
        # Recalculate fertility score based on total tokens and words
        grouped['fertility_score'] = grouped['total_tokens'] / grouped['total_words']
        
        # Sort by fertility score (lower is better)
        sorted_results = grouped.sort_values(by='fertility_score')
        
        # Print leaderboard
        print("\n" + "="*20)
        print("ARABIC TOKENIZER LEADERBOARD".center(20))
        print("="*20)
        print(f"{'👳 Tashkeel':<12} {'📛 Model':<30} {'🪺 Fertility':<12} {'➕ Total Tokens':<15} {'📘 Vocab Size':<12} {'Tokenizer Class':<25}")
        print("-"*20)
        
        for _, row in sorted_results.iterrows():
            print(f"{row['preserves_tashkeel']:<12} {row['model_name'][:28]:<30} {row['fertility_score']:.4f}      {row['total_tokens']:,}      {row['vocab_size'] or 'N/A':<12} {row['tokenizer_class'][:25]}")
            
        print("="*20)
        print(f"Evaluated on {len(self.datasets)} datasets with {len(self.tokenizers)} tokenizers")
        print("="*20)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Arabic tokenizers")
    parser.add_argument("--custom", help="Path to custom tokenizer")
    parser.add_argument("--output", default="tokenizer_results.csv", help="Output file for results")
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = TokenizerEvaluator(
        custom_tokenizer_path=args.custom)
    
    # Run evaluation
    evaluator.run_evaluation()
    
    # Print leaderboard
    evaluator.print_leaderboard()
    
    # Save results
    evaluator.save_results(args.output)


if __name__ == "__main__":
    main()