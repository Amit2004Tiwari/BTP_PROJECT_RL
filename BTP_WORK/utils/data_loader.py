"""
Deep Reinforcement Learning Classifier for Book Review Content Detection
High-Accuracy Data Loader with Advanced Preprocessing

Author: Research Project
Python Version: 3.12.11
System: Linux (SSH Compatible)
Optimized for Maximum Accuracy and Performance
"""

import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional, Union
from tqdm.auto import tqdm
import logging
from collections import defaultdict
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings('ignore')
from typing import List, Dict, Tuple, Any
# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import config
from sentence_transformers import util

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedDataLoader:
    """
    High-Performance Data Loader for Book Review Classification
    Optimized for maximum accuracy and research-grade quality
    """
    
    def __init__(self, dataset_path: str = None):
        """
        Initialize the Advanced Data Loader
        
        Args:
            dataset_path: Path to the book reviews CSV file
        """
        self.dataset_path = dataset_path or config.DATASET_PATH
        self.device = config.DEVICE
        self.config = config
        
        # Initialize sentence transformer for highest accuracy
        logger.info(f"Loading sentence transformer: {config.PRIMARY_ENCODER}")
        self.sentence_encoder = SentenceTransformer(
            config.PRIMARY_ENCODER,
            device=self.device,
            
        )
        
        # Initialize tokenizer for advanced text processing
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.PRIMARY_ENCODER
        )
        
        # Cache for preprocessed data
        self.cache_dir = config.PROJECT_ROOT / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Data containers
        self.raw_data = None
        self.processed_data = None
        self.genre_vocab = {}
        self.stats = {}
        
        logger.info("✅ Advanced Data Loader initialized successfully")
    
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw dataset with comprehensive validation and error handling
        
        Returns:
            DataFrame: Loaded and validated dataset
        """
        logger.info(f"Loading dataset from: {self.dataset_path}")
        
        try:
            # Load with optimized settings for large datasets
            self.raw_data = pd.read_csv(
                self.dataset_path,
                encoding='utf-8',
                low_memory=False,
                na_values=['', 'NaN', 'null', 'NULL'],
                dtype={
                    'ISBN': str,
                    'total_reviews': float,
                    'fiction_nonfiction': str,
                    'categories': str,
                    'description_text': str,
                    'review_text': str
                }
            )
            
            logger.info(f"✅ Dataset loaded successfully: {len(self.raw_data)} rows")
            
            # Validate required columns
            required_columns = [
                'ISBN', 'description_text', 'review_text', 
                'fiction_nonfiction', 'categories'
            ]
            
            missing_columns = [col for col in required_columns if col not in self.raw_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Basic data quality checks
            self._validate_data_quality()
            
            return self.raw_data
            
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {str(e)}")
            raise
    
    def _validate_data_quality(self):
        """
        Comprehensive data quality validation for maximum accuracy
        """
        logger.info("Performing comprehensive data quality validation...")
        
        # Check for null values in critical columns
        null_counts = self.raw_data[['description_text', 'review_text']].isnull().sum()
        if null_counts.any():
            logger.warning(f"Found null values: {null_counts.to_dict()}")
        
        # Remove rows with missing critical data
        initial_count = len(self.raw_data)
        self.raw_data = self.raw_data.dropna(subset=['description_text', 'review_text'])
        self.raw_data = self.raw_data.reset_index(drop=True)
        final_count = len(self.raw_data)
        
        if initial_count != final_count:
            logger.info(f"Removed {initial_count - final_count} rows with missing data")
        
        # Validate text lengths
        desc_lengths = self.raw_data['description_text'].str.len()
        review_lengths = self.raw_data['review_text'].str.len()
        
        self.stats = {
            'total_books': len(self.raw_data),
            'avg_description_length': desc_lengths.mean(),
            'avg_review_length': review_lengths.mean(),
            'min_description_length': desc_lengths.min(),
            'max_description_length': desc_lengths.max(),
            'min_review_length': review_lengths.min(),
            'max_review_length': review_lengths.max()
        }
        
        logger.info(f"✅ Data quality validation completed: {self.stats}")
    
    def advanced_text_preprocessing(self, text: str) -> str:
        """
        Advanced text preprocessing for maximum accuracy
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            str: Cleaned and preprocessed text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string and handle encoding issues
        text = str(text)
        
        # Advanced text cleaning pipeline
        # 1. Fix encoding issues
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        # 2. Normalize unicode characters
        import unicodedata
        text = unicodedata.normalize('NFKD', text)
        
        # 3. Remove excessive whitespace while preserving sentence structure
        text = re.sub(r'\s+', ' ', text)
        
        # 4. Fix common punctuation issues
        text = re.sub(r'([.!?])\1+', r'\1', text)  # Multiple punctuation
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Missing space after punctuation
        
        # 5. Handle contractions properly
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # 6. Remove HTML entities and tags
        import html
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', '', text)
        
        # 7. Normalize quotes and apostrophes
        text = re.sub(r"['']", "'", text)  # Use double quotes outside
        text = re.sub(r'["""]', '"', text)  # This one should already work

        
        # 8. Final cleanup
        text = text.strip()
        
        return text
    
    def intelligent_sentence_splitting(self, text: str) -> List[str]:
        """
        Intelligent sentence splitting optimized for book reviews
        
        Args:
            text: Text to split into sentences
            
        Returns:
            List[str]: List of properly split sentences
        """
        if not text or pd.isna(text):
            return []
        
        # Use advanced sentence boundary detection
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        from nltk.tokenize import sent_tokenize
        
        # Initial sentence splitting
        sentences = sent_tokenize(text)
        
        # Advanced post-processing for book reviews
        cleaned_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Filter out very short or very long sentences
            if (len(sentence) < config.MIN_SENTENCE_LENGTH or 
                len(sentence) > config.MAX_SENTENCE_LENGTH):
                continue
            
            # Remove sentences that are mostly punctuation or numbers
            if re.match(r'^[^a-zA-Z]*$', sentence):
                continue
            
            # Handle special cases in book reviews
            # Skip rating-only sentences
            if re.match(r'^\d+(\.\d+)?[/\s]*\d*\s*(stars?|out of|rating).*$', sentence.lower()):
                continue
            
            # Skip very repetitive sentences
            word_count = len(sentence.split())
            unique_words = len(set(sentence.lower().split()))
            if word_count > 0 and unique_words / word_count < 0.5:
                continue
            
            cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def extract_and_process_genres(self, categories: str, fiction_nonfiction: str) -> List[str]:
        """
        Extract and process genre information for enhanced accuracy
        
        Args:
            categories: Raw categories string
            fiction_nonfiction: Fiction/non-fiction classification
            
        Returns:
            List[str]: Processed genre list
        """
        genres = []
        
        # Add fiction/non-fiction
        if pd.notna(fiction_nonfiction):
            genres.append(fiction_nonfiction.lower().strip())
        
        # Process categories
        if pd.notna(categories):
            # Split by common delimiters
            category_list = re.split(r'[&,;|]', str(categories))
            
            for category in category_list:
                category = category.strip().lower()
                if category and len(category) > 2:  # Filter out very short categories
                    # Normalize common genre variations
                    category = self._normalize_genre(category)
                    if category not in genres:
                        genres.append(category)
        
        # If no genres were found, fallback to 'unknown'
        if not genres:
            genres = ['unknown']
        
        return genres[:config.DataConfig.MAX_GENRES_PER_BOOK]  # Limit number of genres

    
    def _normalize_genre(self, genre: str) -> str:
        """
        Normalize genre strings for consistency
        
        Args:
            genre: Raw genre string
            
        Returns:
            str: Normalized genre
        """
        # Common genre normalizations for book reviews
        normalizations = {
            'sci-fi': 'science fiction',
            'scifi': 'science fiction',
            'fantasy': 'fantasy',
            'romance': 'romance',
            'thriller': 'thriller',
            'mystery': 'mystery',
            'horror': 'horror',
            'biography': 'biography',
            'autobiography': 'biography',
            'memoir': 'biography',
            'history': 'history',
            'historical': 'historical fiction',
            'contemporary': 'contemporary fiction',
            'young adult': 'young adult',
            'ya': 'young adult',
            'children': 'children',
            'childrens': 'children'
        }
        
        return normalizations.get(genre, genre)
    
    def calculate_semantic_similarity(self, sentence_emb, description_emb, genre_emb, sentence_text: str, genres: List[str]) -> Dict[str, float]:
        """
        Compute semantic similarity features using precomputed embeddings
        AND compute linguistic/content indicators (plot keywords, lengths, etc).
        Optimized with vectorised similarity (no .item() in loops).
        """

        # --- Ensure all embeddings are 2D for vector ops ---
        if sentence_emb.dim() == 1:
            sentence_emb = sentence_emb.unsqueeze(0)
        if description_emb.dim() == 1:
            description_emb = description_emb.unsqueeze(0)
        if genre_emb.dim() == 1:
            genre_emb = genre_emb.unsqueeze(0)

        # --- Fast batched similarity computation ---
        sims = util.cos_sim(sentence_emb, torch.cat([description_emb, genre_emb], dim=0))
        cosine_sim = sims[0, 0].detach().cpu().numpy().astype(float)
        genre_sim = sims[0, 1].detach().cpu().numpy().astype(float)

        # --- Extra linguistic / rule-based features ---
        genre_relevance = self._calculate_genre_relevance(sentence_text, genres)
        content_indicators = self._analyze_content_indicators(sentence_text)

        # Handle length features gracefully
        if not hasattr(self, 'dummy_desc'):
            self.dummy_desc = self.advanced_text_preprocessing(" ")
        length_features = self._calculate_length_features(sentence_text, self.dummy_desc)

        # --- Build final features dict ---
        features = {
            'cosine_similarity': cosine_sim,
            'genre_similarity': genre_sim,
            'genre_relevance': float(genre_relevance),
            'plot_keywords': float(content_indicators['plot_keywords']),
            'character_mentions': float(content_indicators['character_mentions']),
            'narrative_indicators': float(content_indicators['narrative_indicators']),
            'length_ratio': float(length_features.get('length_ratio', 0.0)),
            'complexity_score': float(length_features.get('complexity_score', 0.0))
        }

        return features



    
    def _calculate_genre_relevance(self, sentence: str, genres: List[str]) -> float:
        """
        Calculate how relevant the sentence is to the book's genres
        """
        if not genres:
            return 0.0
        
        sentence_lower = sentence.lower()
        genre_matches = 0
        
        # Genre-specific keywords
        genre_keywords = {
            'mystery': ['mystery', 'detective', 'murder', 'crime', 'investigation', 'clues'],
            'romance': ['love', 'relationship', 'romance', 'heart', 'passion', 'romantic'],
            'fantasy': ['magic', 'fantasy', 'dragon', 'wizard', 'magical', 'mythical'],
            'science fiction': ['space', 'future', 'technology', 'alien', 'robot', 'sci-fi'],
            'thriller': ['suspense', 'thriller', 'danger', 'tension', 'chase', 'intense'],
            'horror': ['horror', 'scary', 'frightening', 'terrifying', 'nightmare', 'fear']
        }
        
        for genre in genres:
            if genre in genre_keywords:
                keywords = genre_keywords[genre]
                for keyword in keywords:
                    if keyword in sentence_lower:
                        genre_matches += 1
                        break
        
        return min(genre_matches / len(genres), 1.0)
    
    def _analyze_content_indicators(self, sentence: str) -> Dict[str, float]:
        """
        Analyze sentence for content-specific indicators
        """
        sentence_lower = sentence.lower()
        
        # Plot-related keywords
        plot_keywords = ['plot', 'story', 'narrative', 'storyline', 'events', 'happens', 
                        'chapter', 'beginning', 'ending', 'climax', 'twist']
        plot_score = sum(1 for keyword in plot_keywords if keyword in sentence_lower)
        
        # Character mentions (common patterns)
        character_patterns = [
            r'\b(he|she|they)\s+(said|did|went|was|were|had)',
            r'\b(protagonist|character|hero|heroine|main character)',
            r'\b[A-Z][a-z]+\s+(said|did|went|was|were|had)'
        ]
        character_score = sum(1 for pattern in character_patterns 
                            if re.search(pattern, sentence))
        
        # Narrative indicators
        narrative_keywords = ['when', 'then', 'after', 'before', 'during', 'while', 
                            'suddenly', 'meanwhile', 'later', 'eventually']
        narrative_score = sum(1 for keyword in narrative_keywords if keyword in sentence_lower)
        
        return {
            'plot_keywords': min(plot_score / len(plot_keywords), 1.0),
            'character_mentions': min(character_score / 3, 1.0),
            'narrative_indicators': min(narrative_score / len(narrative_keywords), 1.0)
        }
    
    def _calculate_length_features(self, sentence: str, description: str) -> Dict[str, float]:
        """
        Calculate length-based features for content detection
        """
        sentence_words = len(sentence.split())
        description_words = len(description.split())
        
        # Length ratio
        length_ratio = sentence_words / max(description_words, 1)
        
        # Complexity score (based on average word length and punctuation)
        avg_word_length = np.mean([len(word) for word in sentence.split()])
        punctuation_count = sum(1 for char in sentence if char in '.!?;:,')
        complexity_score = (avg_word_length + punctuation_count / len(sentence)) / 10
        
        return {
            'length_ratio': min(length_ratio, 2.0),
            'complexity_score': min(complexity_score, 1.0)
        }
    
    def process_dataset_for_rl(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process the entire dataset for RL training with maximum accuracy
        Optimized for speed (vectorized + GPU batching where possible).
        Fixed: Removes double encoding, adds validation, uses precomputed embeddings efficiently.
        """
        logger.info("🚀 Starting comprehensive dataset processing for RL training...")
        
        cache_file = self.cache_dir / "processed_rl_samples_with_embeddings.pkl"
        
        # Check if cached version exists
        if cache_file.exists():
            logger.info("Found advanced cached data with embeddings, loading...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        if self.raw_data is None:
            self.load_raw_data()
        
        # --- Load precomputed embeddings (CPU or GPU) ---
        logger.info("Loading precomputed embeddings for data processing...")
        embeddings_dir = config.PROJECT_ROOT / "data" / "embeddings"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            description_embeddings = torch.load(embeddings_dir / "description_embeddings.pt", map_location=device)
            review_embeddings = torch.load(embeddings_dir / "review_embeddings.pt", map_location=device)
            logger.info(f"✅ Precomputed embeddings loaded into {device.upper()} memory.")
        except FileNotFoundError:
            logger.error("❌ Precomputed embeddings not found! Please run precompute_embeddings.py first.")
            raise

        # ✅ ADDED: Validate embedding dimensions and counts
        logger.info("🔍 Validating precomputed embeddings...")
        expected_dim = config.EMBEDDING_DIMENSION
        
        if description_embeddings.shape[1] != expected_dim:
            raise ValueError(f"Description embedding dimension {description_embeddings.shape[1]} != expected {expected_dim}")
        
        if review_embeddings.shape[1] != expected_dim:
            raise ValueError(f"Review embedding dimension {review_embeddings.shape[1]} != expected {expected_dim}")
        
        if description_embeddings.shape[0] != len(self.raw_data):
            raise ValueError(f"Description embedding count {description_embeddings.shape[0]} != dataset size {len(self.raw_data)}")
        
        if review_embeddings.shape[0] != len(self.raw_data):
            raise ValueError(f"Review embedding count {review_embeddings.shape[0]} != dataset size {len(self.raw_data)}")
        
        logger.info(f"✅ Embedding validation passed:")
        logger.info(f"   📊 Description embeddings: {description_embeddings.shape}")
        logger.info(f"   📊 Review embeddings: {review_embeddings.shape}")
        logger.info(f"   📊 Expected dimension: {expected_dim}")

        processed_samples = []
        total_books = len(self.raw_data)

        # ⚡️ Pre-encode ALL genre texts in batch
        logger.info("🔄 Pre-encoding genre embeddings...")
        genre_texts = []
        for _, row in self.raw_data.iterrows():
            genres = self.extract_and_process_genres(row['categories'], row['fiction_nonfiction'])
            genre_text = " ".join(genres) if genres else "unknown"
            genre_texts.append(genre_text)
        
        genre_embeddings = self.sentence_encoder.encode(
            genre_texts, convert_to_tensor=True, device=device, show_progress_bar=True, batch_size=64
        )

        # ⚡️ Process each book efficiently
        with tqdm(total=total_books, desc="Processing books and attaching embeddings") as pbar:
            for idx, row in self.raw_data.iterrows():
                book_data = {
                    'isbn': row['ISBN'] if 'ISBN' in row else None,
                    'description': self.advanced_text_preprocessing(row['description_text']),
                }

                review_text = self.advanced_text_preprocessing(row['review_text'])
                sentences = self.intelligent_sentence_splitting(review_text)

                if not sentences:
                    pbar.update(1)
                    continue

                if len(sentences) > config.DataConfig.MAX_SENTENCES_PER_REVIEW:
                    sentences = sentences[:config.DataConfig.MAX_SENTENCES_PER_REVIEW]

                # ✅ FIXED: Use precomputed embeddings directly (no re-encoding)
                desc_emb = description_embeddings[idx]
                review_emb = review_embeddings[idx]  # Use this as sentence embedding
                genre_emb = genre_embeddings[idx]

                # Process each sentence using precomputed review embedding
                for sentence_idx, sentence in enumerate(sentences):
                    if len(sentence.split()) < 3:
                        continue

                    genres = self.extract_and_process_genres(row['categories'], row['fiction_nonfiction'])
                    similarity_features = self.calculate_semantic_similarity(
                        review_emb, desc_emb, genre_emb, sentence, genres
                    )

                    processed_samples.append({
                        "isbn": book_data['isbn'],
                        "sentence_embedding": review_emb.detach().cpu().float(),
                        "description_embedding": desc_emb.detach().cpu().float(),
                        "genre_embedding": genre_emb.detach().cpu().float(),
                        "features": similarity_features,
                        "label": 1 if similarity_features["cosine_similarity"] >= config.EnvironmentConfig.SIMILARITY_THRESHOLD_HIGH else 0,
                        "confidence": abs(similarity_features["cosine_similarity"] - 0.5) * 2,
                        "sentence_text": sentence,
                        "genres": genres
                    })

                pbar.update(1)

        # Calculate metadata
        metadata = {
            'total_samples': len(processed_samples),
            'total_books': total_books,
            'avg_sentences_per_book': len(processed_samples) / total_books if total_books > 0 else 0,
            'content_ratio': sum(1 for s in processed_samples if s['label'] == 1) / len(processed_samples) if processed_samples else 0,
            'genre_distribution': self._calculate_genre_distribution(processed_samples),
            'processing_stats': self.stats,
            'embedding_stats': {
                'description_shape': list(description_embeddings.shape),
                'review_shape': list(review_embeddings.shape),
                'genre_shape': list(genre_embeddings.shape),
                'embedding_dimension': expected_dim,
                'device_used': device
            }
        }

        # Cache the processed data
        logger.info(f"💾 Caching processed data with embeddings to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump((processed_samples, metadata), f)

        logger.info(f"✅ Dataset processing completed: {len(processed_samples)} samples")
        logger.info(f"📊 Content/Non-content ratio: {metadata['content_ratio']:.3f}")
        logger.info(f"⚡ Performance: No redundant encoding - using precomputed embeddings only!")
        
        return processed_samples, metadata


    def _calculate_genre_distribution(self, samples: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of genres in the dataset"""
        genre_counts = defaultdict(int)
        for sample in samples:
            for genre in sample['genres']:
                genre_counts[genre] += 1
        return dict(genre_counts)
    
    def create_train_val_test_splits(self, processed_samples: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Create stratified train/validation/test splits for maximum accuracy
        
        Args:
            processed_samples: List of processed samples
            
        Returns:
            Dict[str, List[Dict]]: Split datasets
        """
        logger.info("🔄 Creating stratified train/validation/test splits...")
        
        # Extract labels for stratification
        labels = [sample['label'] for sample in processed_samples]
        
        # First split: train + val, test
        train_val_samples, test_samples, train_val_labels, test_labels = train_test_split(
            processed_samples, labels,
            test_size=config.DataConfig.TEST_SPLIT,
            stratify=labels,
            random_state=config.RANDOM_SEED
        )
        
        # Second split: train, val
        val_size = config.DataConfig.VALIDATION_SPLIT / (config.DataConfig.TRAIN_SPLIT + config.DataConfig.VALIDATION_SPLIT)
        train_samples, val_samples, _, _ = train_test_split(
            train_val_samples, train_val_labels,
            test_size=val_size,
            stratify=train_val_labels,
            random_state=config.RANDOM_SEED
        )
        
        splits = {
            'train': train_samples,
            'validation': val_samples,
            'test': test_samples
        }
        
        # Log split statistics
        for split_name, split_data in splits.items():
            content_count = sum(1 for s in split_data if s['label'] == 1)
            total_count = len(split_data)
            logger.info(f"{split_name}: {total_count} samples, "
                       f"{content_count/total_count:.3f} content ratio")
        
        return splits
    
    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Create optimized DataLoaders for training, validation, and testing
        
        Returns:
            Dict[str, DataLoader]: DataLoaders for each split
        """
        logger.info("🔧 Creating optimized DataLoaders...")
        
        # Process dataset if not already done
        processed_samples, metadata = self.process_dataset_for_rl()
        splits = self.create_train_val_test_splits(processed_samples)
        
        dataloaders = {}
        
        for split_name, split_data in splits.items():
            dataset = BookReviewDataset(split_data, self.sentence_encoder)
            
            # Optimize DataLoader settings based on split
            batch_size = config.DQNConfig.BATCH_SIZE if split_name == 'train' else config.EvaluationConfig.EVALUATION_BATCH_SIZE
            shuffle = split_name == 'train'
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=config.NUM_WORKERS,
                pin_memory=config.PIN_MEMORY,
                persistent_workers=config.PERSISTENT_WORKERS,
                drop_last=split_name == 'train'  # Only drop last for training
            )
            
            dataloaders[split_name] = dataloader
        
        logger.info(f"✅ DataLoaders created successfully")
        return dataloaders, metadata

class BookReviewDataset(Dataset):
    """
    High-Performance PyTorch Dataset for Book Review Classification
    OPTIMIZED: Uses pre-stored embeddings to avoid on-the-fly calculation.
    """
    
    def __init__(self, samples: List, sentence_encoder: SentenceTransformer = None):
        """
        Initialize the dataset
        
        Args:
            samples: List of processed samples with pre-stored embeddings
            sentence_encoder: No longer used here, kept for API consistency
        """
        self.samples = samples
        self.device = config.DEVICE
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Embeddings are already stored as torch tensors
        sentence_embedding = sample['sentence_embedding']
        description_embedding = sample['description_embedding']
        genre_embedding = sample['genre_embedding']

        # Concatenate to form state
        state = torch.cat([sentence_embedding, description_embedding, genre_embedding], dim=0).to(self.device)

        # Get features safely
        features_dict = sample.get('features', {})
        
        def safe_float(x, default=0.0):
            if x is None:
                return default
            if isinstance(x, (float, int)):
                return float(x)
            if hasattr(x, 'item'):
                return float(x.item())
            try:
                return float(x)
            except Exception:
                return default

        features = torch.tensor([
            safe_float(features_dict.get('cosine_similarity', 0.0)),
            safe_float(features_dict.get('genre_relevance', 0.0)),
            safe_float(features_dict.get('plot_keywords', 0.0)),
            safe_float(features_dict.get('character_mentions', 0.0)),
            safe_float(features_dict.get('narrative_indicators', 0.0)),
            safe_float(features_dict.get('length_ratio', 0.0)),
            safe_float(features_dict.get('complexity_score', 0.0))
        ], dtype=torch.float32, device=self.device)

        # ✅ CRITICAL: MUST RETURN THE DICTIONARY
        return {
            'state': state,
            'features_tensor': features,
            'features': features_dict,
            'label': torch.tensor(sample['label'], dtype=torch.long, device=self.device),
            'confidence': torch.tensor(sample.get('confidence', 0.0), dtype=torch.float32, device=self.device),
            'sentence': sample.get('sentence_text', ''),
            'isbn': sample.get('isbn', None)
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def initialize_data_loader(dataset_path: str = None) -> AdvancedDataLoader:
    """
    Initialize the data loader with validation
    
    Args:
        dataset_path: Optional path to dataset
        
    Returns:
        AdvancedDataLoader: Initialized data loader
    """
    logger.info("🎯 Initializing Advanced Data Loader for maximum accuracy...")
    
    data_loader = AdvancedDataLoader(dataset_path)
    
    # Validate configuration
    from config import validate_config
    validate_config()
    
    logger.info("✅ Data Loader initialization completed successfully")
    return data_loader

def get_dataset_statistics(data_loader: AdvancedDataLoader) -> Dict:
    """
    Get comprehensive dataset statistics for research analysis
    
    Args:
        data_loader: Initialized data loader
        
    Returns:
        Dict: Comprehensive statistics
    """
    if data_loader.raw_data is None:
        data_loader.load_raw_data()
    logger.info("Ankit")
    return {
        'basic_stats': data_loader.stats,
        'data_quality': {
            'null_percentages': data_loader.raw_data.isnull().sum() / len(data_loader.raw_data),
            'duplicate_count': data_loader.raw_data.duplicated().sum(),
            'unique_books': data_loader.raw_data['ISBN'].nunique()
        }
    }

if __name__ == "__main__":
    # Test the data loader
    logger.info("🧪 Testing Advanced Data Loader...")
    
    data_loader = initialize_data_loader()
    dataloaders, metadata = data_loader.get_dataloaders()
    
    logger.info(f"✅ Test completed successfully!")
    logger.info(f"📊 Metadata: {metadata}")
