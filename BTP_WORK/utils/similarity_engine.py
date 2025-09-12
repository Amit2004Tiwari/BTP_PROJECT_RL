"""
Ultra-Fast Semantic Similarity Engine Using Precomputed Embeddings
Maximum Speed + Maximum Accuracy for Book Review Analysis

Author: Research Project
Python Version: 3.12.11
Optimized for Lightning-Fast Training with Precomputed Data
"""

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)

from pathlib import Path
import pickle
from collections import defaultdict
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS not available. Install with: pip install faiss-cpu")
    FAISS_AVAILABLE = False

from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import config

class FastSemanticSimilarityEngine:
    """
    Ultra-Fast Semantic Similarity Engine Using Precomputed Embeddings
    Designed for maximum speed while preserving research-grade accuracy
    """
    
    def __init__(self, embeddings_dir: Union[str, Path] = None, fallback_to_compute: bool = True):
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else config.PROJECT_ROOT / "data" / "embeddings"

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fallback_to_compute = fallback_to_compute
        
        print(f"🚀 Initializing FastSemanticSimilarityEngine...")
        print(f"📁 Embeddings directory: {self.embeddings_dir}")
        
        # Load precomputed embeddings
        self.precomputed_loaded = self._load_precomputed_embeddings()
        
        # Initialize fallback model if needed
        self.fallback_model = None
        if self.fallback_to_compute and not self.precomputed_loaded:
            print("⚠️  Precomputed embeddings not found, initializing fallback model...")
            self.fallback_model = SentenceTransformer(
                config.PRIMARY_ENCODER,
                device=self.device
            )
            self.embedding_dim = self.fallback_model.get_sentence_embedding_dimension()
        
        # Performance caches
        self.similarity_cache = {}
        
        print(f"✅ FastSemanticSimilarityEngine initialized")
        if self.precomputed_loaded:
            print(f"🎯 Using precomputed embeddings: {self.metadata['num_samples']:,} samples")
            print(f"📊 Embedding dimension: {self.metadata['embedding_dim']}")
        else:
            print(f"⚡ Using fallback live computation")
    
    def _load_precomputed_embeddings(self) -> bool:
        """Load precomputed embeddings from disk"""
        try:
            # Check if all required files exist
            required_files = [
                "description_embeddings.pt",
                "review_embeddings.pt", 
                "metadata.pkl"
            ]
            
            for file_name in required_files:
                if not (self.embeddings_dir / file_name).exists():
                    print(f"❌ Missing precomputed file: {file_name}")
                    return False
            
            print("📊 Loading precomputed embeddings...")
            
            # Load embeddings
            self.description_embeddings = torch.load(
                self.embeddings_dir / "description_embeddings.pt",
                map_location=self.device
            )
            self.review_embeddings = torch.load(
                self.embeddings_dir / "review_embeddings.pt",
                map_location=self.device
            )
            
            # Load metadata
            with open(self.embeddings_dir / "metadata.pkl", 'rb') as f:
                self.metadata = pickle.load(f)
            
            # Load index mapping if available
            if (self.embeddings_dir / "index_mapping.pkl").exists():
                with open(self.embeddings_dir / "index_mapping.pkl", 'rb') as f:
                    self.index_mapping = pickle.load(f)
            else:
                self.index_mapping = None
            
            self.embedding_dim = self.metadata['embedding_dim']
            
            print(f"✅ Loaded {self.metadata['num_samples']:,} precomputed embeddings")
            print(f"📊 Description embeddings: {self.description_embeddings.shape}")
            print(f"📊 Review embeddings: {self.review_embeddings.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load precomputed embeddings: {e}")
            return False
    
    def calculate_similarity_by_index(self, review_idx: int, description_idx: int) -> float:
        """
        Calculate similarity using precomputed embeddings by index (ULTRA FAST)
        
        Args:
            review_idx: Index of review embedding
            description_idx: Index of description embedding
            
        Returns:
            float: Cosine similarity score
        """
        if not self.precomputed_loaded:
            raise ValueError("Precomputed embeddings not loaded. Cannot use index-based similarity.")
        
        # Check cache first
        cache_key = (review_idx, description_idx)
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Get precomputed embeddings (no computation needed!)
        review_emb = self.review_embeddings[review_idx]
        desc_emb = self.description_embeddings[description_idx]
        
        # Fast cosine similarity (embeddings are already normalized)
        similarity = torch.dot(review_emb, desc_emb).item()
        
        # Cache result
        self.similarity_cache[cache_key] = similarity
        
        return similarity
    
    def batch_calculate_similarities(self, review_indices: List[int], 
                                   description_indices: List[int]) -> np.ndarray:
        """
        Calculate multiple similarities in one batch (EVEN FASTER)
        
        Args:
            review_indices: List of review indices
            description_indices: List of description indices
            
        Returns:
            np.ndarray: Array of similarity scores
        """
        if not self.precomputed_loaded:
            raise ValueError("Precomputed embeddings not loaded.")
        
        # Get batch embeddings
        review_embs = self.review_embeddings[review_indices]
        desc_embs = self.description_embeddings[description_indices]
        
        # Batch cosine similarity
        similarities = torch.sum(review_embs * desc_embs, dim=1)
        
        return similarities.cpu().numpy()
    
    def calculate_advanced_similarity_by_index(self, review_idx: int, 
                                             description_idx: int) -> Dict[str, float]:
        """
        Calculate advanced similarity metrics using precomputed embeddings
        
        Args:
            review_idx: Index of review embedding
            description_idx: Index of description embedding
            
        Returns:
            Dict[str, float]: Dictionary of similarity metrics
        """
        if not self.precomputed_loaded:
            raise ValueError("Precomputed embeddings not loaded.")
        
        # Get precomputed embeddings
        emb1 = self.review_embeddings[review_idx]
        emb2 = self.description_embeddings[description_idx]

        
        # Calculate various similarity metrics
        similarities = {}
        
        # 1. Cosine similarity (fastest with normalized embeddings)
        similarities['cosine'] = float(torch.dot(emb1, emb2))

# Euclidean
        euclidean_dist = torch.norm(emb1 - emb2)
        similarities['euclidean_sim'] = float(1.0 / (1.0 + euclidean_dist))

        # Manhattan
        manhattan_dist = torch.sum(torch.abs(emb1 - emb2))
        similarities['manhattan_sim'] = float(1.0 / (1.0 + manhattan_dist))

        # Dot product
        similarities['dot_product'] = float(torch.dot(emb1, emb2))

        # Pearson (torch-based)
        x = emb1 - emb1.mean()
        y = emb2 - emb2.mean()
        pearson = (x @ y) / (torch.norm(x) * torch.norm(y) + 1e-8)
        similarities['pearson'] = float(pearson)
        if np.isnan(similarities['pearson']):
            similarities['pearson'] = 0.0
        
        return similarities
    
    def analyze_content_similarity_by_index(self, review_idx: int, description_idx: int,
                                          review_text: str = None, description_text: str = None) -> Dict[str, Any]:
        """
        Comprehensive content similarity analysis using precomputed embeddings + text analysis
        
        Args:
            review_idx: Index of review embedding
            description_idx: Index of description embedding
            review_text: Optional review text for lexical analysis
            description_text: Optional description text for lexical analysis
            
        Returns:
            Dict[str, Any]: Comprehensive similarity analysis
        """
        analysis = {}
        
        # 1. Fast similarity using precomputed embeddings
        sim_metrics = self.calculate_advanced_similarity_by_index(review_idx, description_idx)
        analysis['description_similarity'] = sim_metrics
        
        # 2. Text-based analysis if texts provided
        if review_text and description_text:
            # Content indicators analysis
            content_indicators = self._analyze_content_indicators(review_text, description_text)
            analysis['content_indicators'] = content_indicators
            
            # Lexical overlap analysis
            lexical_analysis = self._analyze_lexical_overlap(review_text, description_text)
            analysis['lexical_overlap'] = lexical_analysis
            
            # Semantic density analysis
            semantic_density = self._calculate_semantic_density(review_text)
            analysis['semantic_density'] = semantic_density
        else:
            # Default values when no text provided
            analysis['content_indicators'] = {
                'word_overlap': 0.0,
                'plot_indicators': 0.0,
                'character_indicators': 0.0,
                'setting_indicators': 0.0
            }
            analysis['lexical_overlap'] = {
                'jaccard_similarity': 0.0,
                'overlap_ratio': 0.0,
                'common_words_count': 0
            }
            analysis['semantic_density'] = 0.0
        
        # 3. Final content probability
        content_probability = self._calculate_content_probability(analysis)
        analysis['content_probability'] = content_probability
        
        return analysis
    
    def get_embedding_by_index(self, embedding_type: str, index: int) -> torch.Tensor:
        """
        Get precomputed embedding by index
        
        Args:
            embedding_type: 'review' or 'description'
            index: Index of the embedding
            
        Returns:
            torch.Tensor: The embedding tensor
        """
        if not self.precomputed_loaded:
            raise ValueError("Precomputed embeddings not loaded.")
        
        if embedding_type == 'review':
            return self.review_embeddings[index]
        elif embedding_type == 'description':
            return self.description_embeddings[index]
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")
    
    def find_most_similar_reviews(self, description_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Find most similar reviews to a description using precomputed embeddings
        
        Args:
            description_idx: Index of description
            top_k: Number of top results
            
        Returns:
            List[Tuple[int, float]]: List of (review_index, similarity_score)
        """
        if not self.precomputed_loaded:
            raise ValueError("Precomputed embeddings not loaded.")
        
        # Get description embedding
        desc_emb = self.description_embeddings[description_idx].unsqueeze(0)
        
        # Calculate similarities with all reviews
        similarities = torch.mm(desc_emb, self.review_embeddings.t()).squeeze()
        
        # Get top-k indices
        top_similarities, top_indices = torch.topk(similarities, top_k)
        
        # Format results
        results = []
        for idx, sim in zip(top_indices.cpu().numpy(), top_similarities.cpu().numpy()):
            results.append((int(idx), float(sim)))
        
        return results
    
    def build_faiss_index_from_precomputed(self, embedding_type: str = 'description'):
        """
        Build FAISS index from precomputed embeddings for ultra-fast search
        
        Args:
            embedding_type: 'review' or 'description'
        """
        if not self.precomputed_loaded:
            raise ValueError("Precomputed embeddings not loaded.")
        
        print(f"🏗️  Building FAISS index from precomputed {embedding_type} embeddings...")
        
        if embedding_type == 'description':
            embeddings = self.description_embeddings.cpu().numpy()
        else:
            embeddings = self.review_embeddings.cpu().numpy()
        
        # Create FAISS index
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(embeddings.astype(np.float32))
        self.faiss_embedding_type = embedding_type
        
        print(f"✅ FAISS index built: {self.faiss_index.ntotal} vectors indexed")
    
    def search_similar_fast_precomputed(self, query_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Ultra-fast similarity search using FAISS index with precomputed embeddings
        
        Args:
            query_idx: Index of query embedding
            top_k: Number of results
            
        Returns:
            List[Tuple[int, float]]: Similar indices with scores
        """
        if not hasattr(self, 'faiss_index'):
            raise ValueError("FAISS index not built. Call build_faiss_index_from_precomputed() first.")
        
        # Get query embedding
        if self.faiss_embedding_type == 'description':
            query_emb = self.description_embeddings[query_idx:query_idx+1].cpu().numpy()
        else:
            query_emb = self.review_embeddings[query_idx:query_idx+1].cpu().numpy()
        
        # Search
        scores, indices = self.faiss_index.search(query_emb.astype(np.float32), top_k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append((int(idx), float(score)))
        
        return results
    
    # ============================================================================
    # TEXT ANALYSIS HELPER METHODS (for when text is available)
    # ============================================================================
    
    def _analyze_content_indicators(self, sentence: str, description: str) -> Dict[str, float]:
        """Analyze content-specific indicators"""
        sentence_lower = sentence.lower()
        description_lower = description.lower()
        
        # Extract key terms
        description_words = set(description_lower.split())
        sentence_words = set(sentence_lower.split())
        
        # Calculate word overlap
        common_words = description_words.intersection(sentence_words)
        word_overlap = len(common_words) / max(len(description_words), 1)
        
        # Content-specific keywords
        plot_keywords = ['plot', 'story', 'storyline', 'narrative', 'chapter', 'ending', 'twist']
        character_keywords = ['character', 'protagonist', 'hero', 'heroine', 'villain']
        setting_keywords = ['setting', 'world', 'place', 'location', 'atmosphere']
        
        plot_score = sum(1 for kw in plot_keywords if kw in sentence_lower) / len(plot_keywords)
        character_score = sum(1 for kw in character_keywords if kw in sentence_lower) / len(character_keywords)
        setting_score = sum(1 for kw in setting_keywords if kw in sentence_lower) / len(setting_keywords)
        
        return {
            'word_overlap': word_overlap,
            'plot_indicators': plot_score,
            'character_indicators': character_score,
            'setting_indicators': setting_score
        }
    
    def _analyze_lexical_overlap(self, text1: str, text2: str) -> Dict[str, float]:
        """Analyze lexical overlap between texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard = len(intersection) / len(union) if union else 0.0
        overlap_ratio = len(intersection) / min(len(words1), len(words2)) if min(len(words1), len(words2)) > 0 else 0.0
        
        return {
            'jaccard_similarity': jaccard,
            'overlap_ratio': overlap_ratio,
            'common_words_count': len(intersection)
        }
    
    def _calculate_semantic_density(self, text: str) -> float:
        """Calculate semantic density of text (simplified for speed)"""
        words = text.split()
        if len(words) < 2:
            return 0.0
        
        # Simple heuristic based on word count and variety
        unique_words = len(set(words))
        total_words = len(words)
        
        # Higher density = more unique words relative to total
        density = unique_words / total_words if total_words > 0 else 0.0
        
        return density
    
    def _calculate_content_probability(self, analysis: Dict[str, Any]) -> float:
        """Calculate final content probability based on all analyses"""
        # Weight different factors
        weights = {
            'cosine_similarity': 0.5,      # Higher weight for embedding similarity
            'content_indicators': 0.2,
            'lexical_overlap': 0.15,
            'semantic_density': 0.15
        }
        
        # Extract key metrics
        cosine_sim = analysis['description_similarity'].get('cosine', 0.0)
        
        content_ind = np.mean([
            analysis['content_indicators']['plot_indicators'],
            analysis['content_indicators']['character_indicators'],
            analysis['content_indicators']['setting_indicators']
        ])
        
        lexical_overlap = analysis['lexical_overlap']['jaccard_similarity']
        semantic_density = analysis.get('semantic_density', 0.0)
        
        # Calculate weighted probability
        probability = (
            cosine_sim * weights['cosine_similarity'] +
            content_ind * weights['content_indicators'] +
            lexical_overlap * weights['lexical_overlap'] +
            semantic_density * weights['semantic_density']
        )
        
        return np.clip(probability, 0.0, 1.0)
    
    # ============================================================================
    # FALLBACK METHODS (for compatibility when precomputed not available)
    # ============================================================================
    
    def encode_texts_fallback(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Fallback method to encode texts when precomputed not available"""
        if self.fallback_model is None:
            raise ValueError("No fallback model available and precomputed embeddings not loaded.")
        
        return self.fallback_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    
    def calculate_similarity_fallback(self, text1: str, text2: str) -> float:
        """Fallback method for text-based similarity"""
        embeddings = self.encode_texts_fallback([text1, text2])
        return float(np.dot(embeddings[0], embeddings[1]))


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_fast_similarity_engine(embeddings_dir: str = "data/embeddings") -> FastSemanticSimilarityEngine:
    """
    Quick utility to create fast similarity engine
    
    Args:
        embeddings_dir: Directory with precomputed embeddings
        
    Returns:
        FastSemanticSimilarityEngine: Initialized engine
    """
    return FastSemanticSimilarityEngine(embeddings_dir)


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

# Alias for backward compatibility
SemanticSimilarityEngine = FastSemanticSimilarityEngine


if __name__ == "__main__":
    # Test fast similarity engine
    print("🧪 Testing FastSemanticSimilarityEngine...")
    
    engine = FastSemanticSimilarityEngine()
    
    if engine.precomputed_loaded:
        # Test with precomputed embeddings
        review_idx = 0
        description_idx = 0
        
        # Test fast similarity
        similarity = engine.calculate_similarity_by_index(review_idx, description_idx)
        print(f"✅ Fast similarity calculated: {similarity:.4f}")
        
        # Test batch similarities
        review_indices = [0, 1, 2, 3, 4]
        desc_indices = [0, 1, 2, 3, 4]
        batch_similarities = engine.batch_calculate_similarities(review_indices, desc_indices)
        print(f"✅ Batch similarities: {batch_similarities}")
        
        # Test comprehensive analysis
        analysis = engine.analyze_content_similarity_by_index(review_idx, description_idx)
        print(f"✅ Content analysis: probability = {analysis['content_probability']:.3f}")
        
    else:
        print("⚠️  Precomputed embeddings not found. Run precompute_embeddings.py first.")
    
    print("🎯 Fast similarity engine test completed!")
