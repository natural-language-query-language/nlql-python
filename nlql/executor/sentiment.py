from abc import ABC, abstractmethod
from typing import Set
from ..utils.text_unit import Language

class BaseSentimentAnalyzer(ABC):
    """
    Base class for sentiment analysis.
    Users can implement their own sentiment analyzer by inheriting from this class.
    """
    
    @abstractmethod
    def analyze(self, text: str, language: Language) -> str:
        """
        Analyze the sentiment of given text.
        
        Args:
            text (str): Text to analyze
            language (Language): Language of the text
            
        Returns:
            str: One of 'positive', 'negative', or 'neutral'
        """
        pass

class WordlistSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Default sentiment analyzer using predefined word lists.
    """
    
    def __init__(self, 
                 positive_words: Set[str] = None,
                 negative_words: Set[str] = None):
        """
        Initialize the analyzer with custom word lists.
        
        Args:
            positive_words (Set[str], optional): Custom positive word list
            negative_words (Set[str], optional): Custom negative word list
        """
        # Default English word lists
        self._default_positive_en = {
            'good', 'great', 'excellent', 'happy', 'wonderful', 'fantastic',
            'amazing', 'awesome', 'nice', 'love', 'perfect', 'beautiful'
        }
        
        self._default_negative_en = {
            'bad', 'poor', 'terrible', 'horrible', 'awful', 'sad', 'worst',
            'hate', 'dislike', 'disappointing', 'ugly', 'wrong'
        }
        
        # Default Chinese word lists
        self._default_positive_zh = {
            '好', '棒', '优秀', '快乐', '完美', '精彩', '优质', '喜欢',
            '出色', '优秀', '卓越', '美好', '杰出', '出众', '优异'
        }
        
        self._default_negative_zh = {
            '差', '糟糕', '可怕', '难过', '最差', '讨厌', '失望', '错误',
            '劣质', '不好', '低劣', '恶劣', '坏', '不行', '不满意'
        }
        
        # Use custom word lists if provided
        self.positive_words = positive_words or (
            self._default_positive_en | self._default_positive_zh
        )
        self.negative_words = negative_words or (
            self._default_negative_en | self._default_negative_zh
        )

    def analyze(self, text: str, language: Language) -> str:
        """
        Analyze text sentiment using word lists.
        
        Args:
            text (str): Text to analyze
            language (Language): Language of the text
            
        Returns:
            str: 'positive', 'negative', or 'neutral'
        """
        text = text.lower()
        
        # Count positive and negative words
        pos_count = sum(1 for word in self.positive_words if word.lower() in text)
        neg_count = sum(1 for word in self.negative_words if word.lower() in text)
        
        # Determine sentiment based on word counts
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'