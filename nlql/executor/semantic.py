from abc import ABC, abstractmethod
import numpy as np
from ..utils.text_unit import Language

class BaseTopicAnalyzer(ABC):
    """
    Base class for topic analysis.
    Users can implement their own topic analyzer by inheriting from this class.
    """
    
    @abstractmethod
    def match_topic(self, text: str, topic: str, language: Language) -> bool:
        """
        Check if text matches the given topic.
        
        Args:
            text (str): Text to analyze
            topic (str): Topic to match against
            language (Language): Language of the text
            
        Returns:
            bool: Whether the text matches the topic
        """
        pass

class BaseSemanticMatcher(ABC):
    """
    Base class for semantic similarity matching.
    Users can implement their own semantic matcher by inheriting from this class.
    """
    
    @abstractmethod
    def compute_similarity(self, text1: str, text2: str, language: Language) -> float:
        """
        Compute semantic similarity between two texts.
        
        Args:
            text1 (str): First text
            text2 (str): Second text to compare with
            language (Language): Language of the texts
            
        Returns:
            float: Similarity score between 0 and 1
        """
        pass

class BaseVectorEncoder(ABC):
    """
    Base class for text-to-vector encoding.
    Users can implement their own vector encoder by inheriting from this class.
    """
    
    @abstractmethod
    def encode(self, text: str, language: Language) -> np.ndarray:
        """
        Encode text into a vector.
        
        Args:
            text (str): Text to encode
            language (Language): Language of the text
            
        Returns:
            np.ndarray: Vector representation of the text
        """
        pass

    def compute_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute distance between two vectors.
        
        Args:
            vec1 (np.ndarray): First vector
            vec2 (np.ndarray): Second vector
            
        Returns:
            float: Distance between vectors
        """
        return float(np.linalg.norm(vec1 - vec2))

    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute similarity between two vectors.
        
        Args:
            vec1 (np.ndarray): First vector
            vec2 (np.ndarray): Second vector
            
        Returns:
            float: Similarity score between vectors (0 to 1)
        """
        return float(
            np.dot(vec1, vec2) / 
            (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        )

# Default implementations (very basic, for demonstration)
class SimpleTopicAnalyzer(BaseTopicAnalyzer):
    """Simple topic analyzer using keyword matching."""
    
    def match_topic(self, text: str, topic: str, language: Language) -> bool:
        # Simple implementation using keyword presence
        keywords = topic.lower().split()
        text_lower = text.lower()
        return all(keyword in text_lower for keyword in keywords)

class SimpleSemanticMatcher(BaseSemanticMatcher):
    """Simple semantic matcher using word overlap."""
    
    def compute_similarity(self, text1: str, text2: str, language: Language) -> float:
        # Simple implementation using word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        overlap = words1.intersection(words2)
        return len(overlap) / max(len(words1), len(words2))

class SimpleVectorEncoder(BaseVectorEncoder):
    """Simple vector encoder using word presence."""
    
    def __init__(self, dimension: int = 100):
        """
        Initialize encoder with specified dimension.
        
        Args:
            dimension (int): Dimension of vectors to generate
        """
        self.dimension = dimension
        self._word_vectors = {}

    def _get_word_vector(self, word: str) -> np.ndarray:
        """Get or create a random vector for a word."""
        if word not in self._word_vectors:
            self._word_vectors[word] = np.random.randn(self.dimension)
        return self._word_vectors[word]

    def encode(self, text: str, language: Language) -> np.ndarray:
        # Simple implementation using average of random word vectors
        words = text.lower().split()
        if not words:
            return np.zeros(self.dimension)
        
        vectors = [self._get_word_vector(word) for word in words]
        return np.mean(vectors, axis=0)