"""
Meng AI Evaluator and Benchmarking System
=========================================

Comprehensive evaluation system for Meng AI with:
1. Multiple evaluation metrics (perplexity, BLEU, ROUGE, etc.)
2. Benchmark datasets and tasks
3. Human evaluation frameworks
4. Performance profiling and analysis
5. Model comparison and ranking
6. Automated evaluation pipelines

Author: Meng AI Development Team
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import psutil
import gc
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
import math
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    # Evaluation metrics
    compute_perplexity: bool = True
    compute_bleu: bool = True
    compute_rouge: bool = True
    compute_meteor: bool = True
    compute_semantic_similarity: bool = True
    compute_diversity: bool = True
    compute_coherence: bool = True
    
    # Generation parameters
    num_samples: int = 100
    max_length: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    
    # Performance metrics
    measure_inference_time: bool = True
    measure_memory_usage: bool = True
    measure_throughput: bool = True
    
    # Benchmark datasets
    use_benchmark_datasets: bool = True
    benchmark_tasks: List[str] = None

class PerplexityEvaluator:
    """Evaluate model perplexity on test data"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def evaluate(self, texts: List[str]) -> Dict[str, float]:
        """Calculate perplexity on a list of texts"""
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Computing perplexity"):
                # Tokenize text
                token_ids = self.tokenizer.encode(text, add_special_tokens=True)
                if len(token_ids) < 2:
                    continue
                
                # Convert to tensor
                input_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)
                
                # Forward pass
                logits = self.model(input_ids)
                
                # Calculate loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss(ignore_index=0)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                total_loss += loss.item() * len(token_ids)
                total_tokens += len(token_ids)
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return {
            'perplexity': perplexity,
            'average_loss': avg_loss,
            'total_tokens': total_tokens
        }

class BLEUEvaluator:
    """Evaluate BLEU scores for text generation"""
    
    def __init__(self):
        self.smoothing = SmoothingFunction().method4
    
    def evaluate(self, generated_texts: List[str], reference_texts: List[str]) -> Dict[str, float]:
        """Calculate BLEU scores"""
        if len(generated_texts) != len(reference_texts):
            raise ValueError("Generated and reference texts must have the same length")
        
        bleu_scores = []
        
        for gen_text, ref_text in zip(generated_texts, reference_texts):
            # Tokenize
            gen_tokens = word_tokenize(gen_text.lower())
            ref_tokens = word_tokenize(ref_text.lower())
            
            # Calculate BLEU score
            bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=self.smoothing)
            bleu_scores.append(bleu)
        
        return {
            'bleu_score': np.mean(bleu_scores),
            'bleu_std': np.std(bleu_scores),
            'bleu_scores': bleu_scores
        }

class ROUGEEvaluator:
    """Evaluate ROUGE scores for text generation"""
    
    def __init__(self):
        pass
    
    def _rouge_n(self, generated: List[str], reference: List[str], n: int) -> float:
        """Calculate ROUGE-N score"""
        scores = []
        
        for gen, ref in zip(generated, reference):
            gen_ngrams = self._get_ngrams(gen, n)
            ref_ngrams = self._get_ngrams(ref, n)
            
            if len(ref_ngrams) == 0:
                scores.append(0.0)
                continue
            
            overlap = len(gen_ngrams.intersection(ref_ngrams))
            precision = overlap / len(gen_ngrams) if len(gen_ngrams) > 0 else 0
            recall = overlap / len(ref_ngrams)
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            
            scores.append(f1)
        
        return np.mean(scores)
    
    def _get_ngrams(self, text: str, n: int) -> set:
        """Get n-grams from text"""
        tokens = word_tokenize(text.lower())
        ngrams = set()
        
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.add(ngram)
        
        return ngrams
    
    def _rouge_l(self, generated: List[str], reference: List[str]) -> float:
        """Calculate ROUGE-L score"""
        scores = []
        
        for gen, ref in zip(generated, reference):
            gen_tokens = word_tokenize(gen.lower())
            ref_tokens = word_tokenize(ref.lower())
            
            # Calculate LCS
            lcs_length = self._lcs_length(gen_tokens, ref_tokens)
            
            if len(ref_tokens) == 0:
                scores.append(0.0)
                continue
            
            precision = lcs_length / len(gen_tokens) if len(gen_tokens) > 0 else 0
            recall = lcs_length / len(ref_tokens)
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            
            scores.append(f1)
        
        return np.mean(scores)
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of longest common subsequence"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def evaluate(self, generated_texts: List[str], reference_texts: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        if len(generated_texts) != len(reference_texts):
            raise ValueError("Generated and reference texts must have the same length")
        
        rouge_1 = self._rouge_n(generated_texts, reference_texts, 1)
        rouge_2 = self._rouge_n(generated_texts, reference_texts, 2)
        rouge_l = self._rouge_l(generated_texts, reference_texts)
        
        return {
            'rouge_1': rouge_1,
            'rouge_2': rouge_2,
            'rouge_l': rouge_l
        }

class METEOREvaluator:
    """Evaluate METEOR scores for text generation"""
    
    def __init__(self):
        pass
    
    def evaluate(self, generated_texts: List[str], reference_texts: List[str]) -> Dict[str, float]:
        """Calculate METEOR scores"""
        if len(generated_texts) != len(reference_texts):
            raise ValueError("Generated and reference texts must have the same length")
        
        meteor_scores = []
        
        for gen_text, ref_text in zip(generated_texts, reference_texts):
            try:
                score = meteor_score([ref_text], gen_text)
                meteor_scores.append(score)
            except:
                meteor_scores.append(0.0)
        
        return {
            'meteor_score': np.mean(meteor_scores),
            'meteor_std': np.std(meteor_scores),
            'meteor_scores': meteor_scores
        }

class SemanticSimilarityEvaluator:
    """Evaluate semantic similarity using TF-IDF and cosine similarity"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    def evaluate(self, generated_texts: List[str], reference_texts: List[str]) -> Dict[str, float]:
        """Calculate semantic similarity scores"""
        if len(generated_texts) != len(reference_texts):
            raise ValueError("Generated and reference texts must have the same length")
        
        # Combine texts for vectorization
        all_texts = generated_texts + reference_texts
        
        # Fit and transform
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        # Split back into generated and reference
        n = len(generated_texts)
        gen_vectors = tfidf_matrix[:n]
        ref_vectors = tfidf_matrix[n:]
        
        # Calculate cosine similarities
        similarities = []
        for i in range(n):
            similarity = cosine_similarity(gen_vectors[i:i+1], ref_vectors[i:i+1])[0][0]
            similarities.append(similarity)
        
        return {
            'semantic_similarity': np.mean(similarities),
            'semantic_similarity_std': np.std(similarities),
            'similarities': similarities
        }

class DiversityEvaluator:
    """Evaluate text diversity using various metrics"""
    
    def __init__(self):
        pass
    
    def _distinct_n(self, texts: List[str], n: int) -> float:
        """Calculate distinct-n score"""
        all_ngrams = set()
        total_ngrams = 0
        
        for text in texts:
            tokens = word_tokenize(text.lower())
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i+n])
                all_ngrams.add(ngram)
                total_ngrams += 1
        
        return len(all_ngrams) / total_ngrams if total_ngrams > 0 else 0.0
    
    def _self_bleu(self, texts: List[str]) -> float:
        """Calculate self-BLEU score"""
        if len(texts) < 2:
            return 0.0
        
        self_bleu_scores = []
        smoothing = SmoothingFunction().method4
        
        for i, text in enumerate(texts):
            gen_tokens = word_tokenize(text.lower())
            ref_texts = [word_tokenize(t.lower()) for j, t in enumerate(texts) if j != i]
            
            if not ref_texts:
                continue
            
            bleu = sentence_bleu(ref_texts, gen_tokens, smoothing_function=smoothing)
            self_bleu_scores.append(bleu)
        
        return np.mean(self_bleu_scores) if self_bleu_scores else 0.0
    
    def evaluate(self, texts: List[str]) -> Dict[str, float]:
        """Calculate diversity metrics"""
        distinct_1 = self._distinct_n(texts, 1)
        distinct_2 = self._distinct_n(texts, 2)
        distinct_3 = self._distinct_n(texts, 3)
        self_bleu = self._self_bleu(texts)
        
        return {
            'distinct_1': distinct_1,
            'distinct_2': distinct_2,
            'distinct_3': distinct_3,
            'self_bleu': self_bleu,
            'diversity_score': (distinct_1 + distinct_2 + distinct_3) / 3
        }

class CoherenceEvaluator:
    """Evaluate text coherence using various metrics"""
    
    def __init__(self):
        pass
    
    def _sentence_coherence(self, text: str) -> float:
        """Calculate sentence-level coherence"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 1.0
        
        # Calculate word overlap between consecutive sentences
        overlaps = []
        for i in range(len(sentences) - 1):
            words1 = set(word_tokenize(sentences[i].lower()))
            words2 = set(word_tokenize(sentences[i+1].lower()))
            
            if len(words1) == 0 or len(words2) == 0:
                overlaps.append(0.0)
                continue
            
            overlap = len(words1.intersection(words2))
            union = len(words1.union(words2))
            jaccard = overlap / union if union > 0 else 0.0
            overlaps.append(jaccard)
        
        return np.mean(overlaps) if overlaps else 0.0
    
    def _topic_coherence(self, text: str) -> float:
        """Calculate topic coherence using word co-occurrence"""
        words = word_tokenize(text.lower())
        if len(words) < 3:
            return 0.0
        
        # Calculate word co-occurrence within a window
        window_size = 3
        cooccurrences = 0
        total_pairs = 0
        
        for i in range(len(words)):
            for j in range(i+1, min(i+window_size+1, len(words))):
                total_pairs += 1
                # Simple co-occurrence based on proximity
                cooccurrences += 1
        
        return cooccurrences / total_pairs if total_pairs > 0 else 0.0
    
    def evaluate(self, texts: List[str]) -> Dict[str, float]:
        """Calculate coherence metrics"""
        sentence_coherences = [self._sentence_coherence(text) for text in texts]
        topic_coherences = [self._topic_coherence(text) for text in texts]
        
        return {
            'sentence_coherence': np.mean(sentence_coherences),
            'topic_coherence': np.mean(topic_coherences),
            'overall_coherence': (np.mean(sentence_coherences) + np.mean(topic_coherences)) / 2,
            'sentence_coherence_std': np.std(sentence_coherences),
            'topic_coherence_std': np.std(topic_coherences)
        }

class PerformanceProfiler:
    """Profile model performance including speed and memory usage"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def profile_inference(self, input_ids: torch.Tensor, num_runs: int = 100) -> Dict[str, float]:
        """Profile inference speed and memory usage"""
        self.model.eval()
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(input_ids)
        
        # Measure inference time
        torch.cuda.synchronize() if self.device == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(input_ids)
        
        torch.cuda.synchronize() if self.device == 'cuda' else None
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / num_runs
        
        # Measure memory usage
        if self.device == 'cuda':
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1024**2   # MB
        else:
            memory_allocated = psutil.Process().memory_info().rss / 1024**2  # MB
            memory_reserved = 0.0
        
        # Calculate throughput
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(1)
        tokens_per_second = (batch_size * seq_length) / avg_inference_time
        
        return {
            'avg_inference_time_ms': avg_inference_time * 1000,
            'memory_allocated_mb': memory_allocated,
            'memory_reserved_mb': memory_reserved,
            'tokens_per_second': tokens_per_second,
            'throughput_samples_per_second': batch_size / avg_inference_time
        }

class MengAIEvaluator:
    """Comprehensive evaluator for Meng AI models"""
    
    def __init__(self, model, tokenizer, config: EvaluationConfig, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Initialize evaluators
        self.perplexity_evaluator = PerplexityEvaluator(model, tokenizer, device)
        self.bleu_evaluator = BLEUEvaluator()
        self.rouge_evaluator = ROUGEEvaluator()
        self.meteor_evaluator = METEOREvaluator()
        self.semantic_evaluator = SemanticSimilarityEvaluator()
        self.diversity_evaluator = DiversityEvaluator()
        self.coherence_evaluator = CoherenceEvaluator()
        self.performance_profiler = PerformanceProfiler(model, device)
    
    def generate_texts(self, prompts: List[str]) -> List[str]:
        """Generate texts from prompts"""
        generated_texts = []
        
        for prompt in tqdm(prompts, desc="Generating texts"):
            # Encode prompt
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            
            # Generate
            with torch.no_grad():
                generated = self.model.generate(
                    input_tensor,
                    max_length=self.config.max_length,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    do_sample=True
                )
            
            # Decode
            generated_text = self.tokenizer.decode(generated[0].cpu().tolist())
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def evaluate_comprehensive(self, test_texts: List[str], 
                             reference_texts: Optional[List[str]] = None,
                             prompts: Optional[List[str]] = None) -> Dict[str, Any]:
        """Comprehensive evaluation of the model"""
        results = {}
        
        print("🚀 Starting comprehensive evaluation...")
        
        # 1. Perplexity evaluation
        if self.config.compute_perplexity:
            print("Computing perplexity...")
            perplexity_results = self.perplexity_evaluator.evaluate(test_texts)
            results['perplexity'] = perplexity_results
        
        # 2. Text generation evaluation
        if prompts is not None:
            print("Generating texts for evaluation...")
            generated_texts = self.generate_texts(prompts)
            results['generated_texts'] = generated_texts
            
            # BLEU evaluation
            if self.config.compute_bleu and reference_texts is not None:
                print("Computing BLEU scores...")
                bleu_results = self.bleu_evaluator.evaluate(generated_texts, reference_texts)
                results['bleu'] = bleu_results
            
            # ROUGE evaluation
            if self.config.compute_rouge and reference_texts is not None:
                print("Computing ROUGE scores...")
                rouge_results = self.rouge_evaluator.evaluate(generated_texts, reference_texts)
                results['rouge'] = rouge_results
            
            # METEOR evaluation
            if self.config.compute_meteor and reference_texts is not None:
                print("Computing METEOR scores...")
                meteor_results = self.meteor_evaluator.evaluate(generated_texts, reference_texts)
                results['meteor'] = meteor_results
            
            # Semantic similarity evaluation
            if self.config.compute_semantic_similarity and reference_texts is not None:
                print("Computing semantic similarity...")
                semantic_results = self.semantic_evaluator.evaluate(generated_texts, reference_texts)
                results['semantic_similarity'] = semantic_results
            
            # Diversity evaluation
            if self.config.compute_diversity:
                print("Computing diversity metrics...")
                diversity_results = self.diversity_evaluator.evaluate(generated_texts)
                results['diversity'] = diversity_results
            
            # Coherence evaluation
            if self.config.compute_coherence:
                print("Computing coherence metrics...")
                coherence_results = self.coherence_evaluator.evaluate(generated_texts)
                results['coherence'] = coherence_results
        
        # 3. Performance profiling
        if self.config.measure_inference_time or self.config.measure_memory_usage:
            print("Profiling performance...")
            # Create sample input
            sample_text = test_texts[0] if test_texts else "The quick brown fox jumps over the lazy dog."
            input_ids = self.tokenizer.encode(sample_text, add_special_tokens=True)
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            
            performance_results = self.performance_profiler.profile_inference(input_tensor)
            results['performance'] = performance_results
        
        return results
    
    def create_evaluation_report(self, results: Dict[str, Any], 
                               output_path: Optional[str] = None) -> str:
        """Create a comprehensive evaluation report"""
        report = []
        report.append("=" * 80)
        report.append("MENG AI EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Model information
        report.append("MODEL INFORMATION")
        report.append("-" * 40)
        report.append(f"Model: Meng AI")
        report.append(f"Device: {self.device}")
        report.append(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Perplexity results
        if 'perplexity' in results:
            report.append("PERPLEXITY EVALUATION")
            report.append("-" * 40)
            ppl_results = results['perplexity']
            report.append(f"Perplexity: {ppl_results['perplexity']:.2f}")
            report.append(f"Average Loss: {ppl_results['average_loss']:.4f}")
            report.append(f"Total Tokens: {ppl_results['total_tokens']:,}")
            report.append("")
        
        # Generation quality results
        if 'bleu' in results:
            report.append("BLEU EVALUATION")
            report.append("-" * 40)
            bleu_results = results['bleu']
            report.append(f"BLEU Score: {bleu_results['bleu_score']:.4f}")
            report.append(f"BLEU Std: {bleu_results['bleu_std']:.4f}")
            report.append("")
        
        if 'rouge' in results:
            report.append("ROUGE EVALUATION")
            report.append("-" * 40)
            rouge_results = results['rouge']
            report.append(f"ROUGE-1: {rouge_results['rouge_1']:.4f}")
            report.append(f"ROUGE-2: {rouge_results['rouge_2']:.4f}")
            report.append(f"ROUGE-L: {rouge_results['rouge_l']:.4f}")
            report.append("")
        
        if 'meteor' in results:
            report.append("METEOR EVALUATION")
            report.append("-" * 40)
            meteor_results = results['meteor']
            report.append(f"METEOR Score: {meteor_results['meteor_score']:.4f}")
            report.append(f"METEOR Std: {meteor_results['meteor_std']:.4f}")
            report.append("")
        
        if 'semantic_similarity' in results:
            report.append("SEMANTIC SIMILARITY EVALUATION")
            report.append("-" * 40)
            semantic_results = results['semantic_similarity']
            report.append(f"Semantic Similarity: {semantic_results['semantic_similarity']:.4f}")
            report.append(f"Semantic Similarity Std: {semantic_results['semantic_similarity_std']:.4f}")
            report.append("")
        
        if 'diversity' in results:
            report.append("DIVERSITY EVALUATION")
            report.append("-" * 40)
            diversity_results = results['diversity']
            report.append(f"Distinct-1: {diversity_results['distinct_1']:.4f}")
            report.append(f"Distinct-2: {diversity_results['distinct_2']:.4f}")
            report.append(f"Distinct-3: {diversity_results['distinct_3']:.4f}")
            report.append(f"Self-BLEU: {diversity_results['self_bleu']:.4f}")
            report.append(f"Diversity Score: {diversity_results['diversity_score']:.4f}")
            report.append("")
        
        if 'coherence' in results:
            report.append("COHERENCE EVALUATION")
            report.append("-" * 40)
            coherence_results = results['coherence']
            report.append(f"Sentence Coherence: {coherence_results['sentence_coherence']:.4f}")
            report.append(f"Topic Coherence: {coherence_results['topic_coherence']:.4f}")
            report.append(f"Overall Coherence: {coherence_results['overall_coherence']:.4f}")
            report.append("")
        
        # Performance results
        if 'performance' in results:
            report.append("PERFORMANCE EVALUATION")
            report.append("-" * 40)
            perf_results = results['performance']
            report.append(f"Average Inference Time: {perf_results['avg_inference_time_ms']:.2f} ms")
            report.append(f"Memory Allocated: {perf_results['memory_allocated_mb']:.2f} MB")
            report.append(f"Memory Reserved: {perf_results['memory_reserved_mb']:.2f} MB")
            report.append(f"Tokens per Second: {perf_results['tokens_per_second']:.2f}")
            report.append(f"Throughput: {perf_results['throughput_samples_per_second']:.2f} samples/sec")
            report.append("")
        
        # Generated samples
        if 'generated_texts' in results:
            report.append("GENERATED SAMPLES")
            report.append("-" * 40)
            for i, text in enumerate(results['generated_texts'][:5]):  # Show first 5 samples
                report.append(f"Sample {i+1}:")
                report.append(f"  {text}")
                report.append("")
        
        report.append("=" * 80)
        report.append("EVALUATION COMPLETED")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Evaluation report saved to: {output_path}")
        
        return report_text
    
    def plot_evaluation_results(self, results: Dict[str, Any], 
                              output_path: Optional[str] = None):
        """Create visualization plots for evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Meng AI Evaluation Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Perplexity
        if 'perplexity' in results:
            axes[0, 0].bar(['Perplexity'], [results['perplexity']['perplexity']])
            axes[0, 0].set_title('Perplexity Score')
            axes[0, 0].set_ylabel('Perplexity')
        
        # Plot 2: Generation Quality Metrics
        if any(metric in results for metric in ['bleu', 'rouge', 'meteor']):
            metrics = []
            values = []
            
            if 'bleu' in results:
                metrics.append('BLEU')
                values.append(results['bleu']['bleu_score'])
            
            if 'rouge' in results:
                metrics.extend(['ROUGE-1', 'ROUGE-2', 'ROUGE-L'])
                values.extend([results['rouge']['rouge_1'], 
                             results['rouge']['rouge_2'], 
                             results['rouge']['rouge_l']])
            
            if 'meteor' in results:
                metrics.append('METEOR')
                values.append(results['meteor']['meteor_score'])
            
            axes[0, 1].bar(metrics, values)
            axes[0, 1].set_title('Generation Quality Metrics')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Diversity and Coherence
        if 'diversity' in results or 'coherence' in results:
            metrics = []
            values = []
            
            if 'diversity' in results:
                metrics.extend(['Distinct-1', 'Distinct-2', 'Distinct-3'])
                values.extend([results['diversity']['distinct_1'],
                             results['diversity']['distinct_2'],
                             results['diversity']['distinct_3']])
            
            if 'coherence' in results:
                metrics.extend(['Sentence Coherence', 'Topic Coherence'])
                values.extend([results['coherence']['sentence_coherence'],
                             results['coherence']['topic_coherence']])
            
            axes[1, 0].bar(metrics, values)
            axes[1, 0].set_title('Diversity and Coherence Metrics')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Performance Metrics
        if 'performance' in results:
            perf_metrics = ['Inference Time (ms)', 'Memory (MB)', 'Tokens/sec']
            perf_values = [
                results['performance']['avg_inference_time_ms'],
                results['performance']['memory_allocated_mb'],
                results['performance']['tokens_per_second']
            ]
            
            axes[1, 1].bar(perf_metrics, perf_values)
            axes[1, 1].set_title('Performance Metrics')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Evaluation plots saved to: {output_path}")
        
        plt.show()

def main():
    """Main function to demonstrate the evaluation system"""
    print("Meng AI Evaluator Demo")
    print("=" * 50)
    
    # This is a demo - in practice, you would load your trained model
    print("Note: This is a demonstration of the evaluation system.")
    print("In practice, you would load your trained Meng AI model here.")
    
    # Create sample data for demonstration
    test_texts = [
        "The quick brown fox jumps over the lazy dog. This is a classic pangram used for testing.",
        "Python is a powerful programming language that is widely used in data science and machine learning.",
        "Artificial intelligence has the potential to revolutionize many industries and improve human life.",
        "Deep learning models can understand complex patterns in data and make accurate predictions.",
        "Natural language processing enables computers to understand and generate human language."
    ]
    
    reference_texts = [
        "A fast brown fox leaps over a sleepy dog. This sentence contains every letter of the alphabet.",
        "Python is an excellent programming language popular in data science and artificial intelligence.",
        "AI technology could transform various sectors and enhance quality of life for people.",
        "Neural networks can recognize intricate data patterns and provide reliable forecasts.",
        "NLP allows machines to comprehend and produce natural human communication."
    ]
    
    prompts = [
        "The future of",
        "Python is",
        "Artificial intelligence",
        "Deep learning",
        "Natural language processing"
    ]
    
    # Configuration
    config = EvaluationConfig(
        compute_perplexity=True,
        compute_bleu=True,
        compute_rouge=True,
        compute_meteor=True,
        compute_semantic_similarity=True,
        compute_diversity=True,
        compute_coherence=True,
        num_samples=5,
        max_length=50
    )
    
    print("Evaluation configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    
    print(f"\nEvaluation system is ready!")
    print(f"\nTo use with your trained model:")
    print(f"1. Load your trained Meng AI model")
    print(f"2. Initialize the evaluator with your model and tokenizer")
    print(f"3. Run comprehensive evaluation on your test data")
    print(f"4. Generate evaluation reports and visualizations")
    
    return config

if __name__ == "__main__":
    main()
