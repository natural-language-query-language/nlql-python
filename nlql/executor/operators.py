from typing import Any, Optional, Union, Type, Callable
from enum import Enum

from nlql.executor.semantic import BaseSemanticMatcher, BaseTopicAnalyzer, BaseVectorEncoder, SimpleSemanticMatcher, SimpleTopicAnalyzer, SimpleVectorEncoder
from ..exceptions import OperatorError
from ..lexer.token import TokenType
from ..utils.text_unit import TextUnit

class ComparisonType(Enum):
    """Types of comparison operations."""
    EQUALS = "equals"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"

class BaseOperator:
    """Base class for all operators."""
    
    def __init__(self, field: str = None):
        """
        Initialize the operator.
        
        Args:
            field (str, optional): Field to operate on
        """
        self.field = field

    def __call__(self, unit: TextUnit, *args: Any) -> bool:
        """
        Execute the operator.
        
        Args:
            unit (TextUnit): Text unit to check
            *args: Additional arguments for the operator
            
        Returns:
            bool: Whether the condition is met
        """
        raise NotImplementedError

class ContainsOperator(BaseOperator):
    """Operator that checks if text contains a substring."""
    
    def __call__(self, unit: TextUnit, substring: str) -> bool:
        """Check if the text unit contains the substring."""
        if not isinstance(substring, str):
            raise OperatorError("CONTAINS requires a string argument")
        return substring.lower() in unit.content.lower()

class StartsWithOperator(BaseOperator):
    """Operator that checks if text starts with a substring."""
    
    def __call__(self, unit: TextUnit, prefix: str) -> bool:
        """Check if the text unit starts with the prefix."""
        if not isinstance(prefix, str):
            raise OperatorError("STARTS_WITH requires a string argument")
        return unit.content.lower().startswith(prefix.lower())

class EndsWithOperator(BaseOperator):
    """Operator that checks if text ends with a substring."""
    
    def __call__(self, unit: TextUnit, suffix: str) -> bool:
        """Check if the text unit ends with the suffix."""
        if not isinstance(suffix, str):
            raise OperatorError("ENDS_WITH requires a string argument")
        return unit.content.lower().endswith(suffix.lower())

class LengthOperator(BaseOperator):
    """Operator that compares text length."""
    
    def __init__(self, comparison_type: ComparisonType):
        """
        Initialize the length operator.
        
        Args:
            comparison_type (ComparisonType): Type of comparison to perform
        """
        super().__init__("length")
        self.comparison_type = comparison_type
        
        # Map comparison types to comparison functions
        self._comparisons = {
            ComparisonType.EQUALS: lambda x, y: x == y,
            ComparisonType.GREATER_THAN: lambda x, y: x > y,
            ComparisonType.LESS_THAN: lambda x, y: x < y,
            ComparisonType.GREATER_EQUAL: lambda x, y: x >= y,
            ComparisonType.LESS_EQUAL: lambda x, y: x <= y
        }

    def __call__(self, unit: TextUnit, value: int) -> bool:
        """Compare the text unit's length with the given value."""
        if not isinstance(value, (int, float)):
            raise OperatorError("LENGTH comparison requires a numeric argument")
            
        length = len(unit.content)
        comparison = self._comparisons.get(self.comparison_type)
        
        if not comparison:
            raise OperatorError(f"Unsupported comparison type: {self.comparison_type}")
            
        return comparison(length, value)

class TopicOperator(BaseOperator):
    """Topic matching operator that uses a configurable analyzer."""
    
    def __init__(self, analyzer: Optional[BaseTopicAnalyzer] = None):
        """
        Initialize the topic operator.
        
        Args:
            analyzer (BaseTopicAnalyzer, optional): Custom topic analyzer.
                If None, uses the SimpleTopicAnalyzer.
        """
        super().__init__("topic")
        self.analyzer = analyzer or SimpleTopicAnalyzer()

    def __call__(self, unit: TextUnit, topic: str) -> bool:
        """
        Check if the text unit matches the given topic.
        
        Args:
            unit (TextUnit): Text unit to analyze
            topic (str): Topic to match against
            
        Returns:
            bool: Whether the text matches the topic
        """
        if not isinstance(topic, str):
            raise OperatorError("TOPIC_IS requires a string argument")
            
        return self.analyzer.match_topic(
            unit.content,
            topic,
            unit.language
        )

class SimilarToOperator(BaseOperator):
    """Semantic similarity operator using a configurable matcher."""
    
    def __init__(self, matcher: Optional[BaseSemanticMatcher] = None):
        """
        Initialize the similarity operator.
        
        Args:
            matcher (BaseSemanticMatcher, optional): Custom semantic matcher.
                If None, uses the SimpleSemanticMatcher.
        """
        super().__init__("similar")
        self.matcher = matcher or SimpleSemanticMatcher()

    def __call__(self, unit: TextUnit, target: str, threshold: float = 0.5) -> bool:
        """
        Check if the text unit is semantically similar to the target text.
        
        Args:
            unit (TextUnit): Text unit to analyze
            target (str): Text to compare with
            threshold (float): Minimum similarity score (0 to 1)
            
        Returns:
            bool: Whether the texts are similar enough
            
        Raises:
            OperatorError: If arguments are invalid
        """
        if not isinstance(target, str):
            raise OperatorError("SIMILAR_TO requires a string argument")
        if not isinstance(threshold, (int, float)):
            raise OperatorError("SIMILAR_TO threshold must be a number")
        if not 0 <= threshold <= 1:
            raise OperatorError("SIMILAR_TO threshold must be between 0 and 1")
            
        similarity = self.matcher.compute_similarity(
            unit.content,
            target,
            unit.language
        )
        return similarity >= threshold

class EmbeddingDistanceOperator(BaseOperator):
    """Vector distance operator using a configurable encoder."""
    
    def __init__(self, encoder: Optional[BaseVectorEncoder] = None):
        """
        Initialize the distance operator.
        
        Args:
            encoder (BaseVectorEncoder, optional): Custom vector encoder.
                If None, uses the SimpleVectorEncoder.
        """
        super().__init__("distance")
        self.encoder = encoder or SimpleVectorEncoder()

    def __call__(self, unit: TextUnit, target: str, threshold: float) -> bool:
        """
        Check if the vector distance is within the threshold.
        
        Args:
            unit (TextUnit): Text unit to analyze
            target (str): Text to compare with
            threshold (float): Maximum allowed distance
            
        Returns:
            bool: Whether the distance is within threshold
            
        Raises:
            OperatorError: If arguments are invalid
        """
        if not isinstance(target, str):
            raise OperatorError("EMBEDDING_DISTANCE requires a string argument")
        if not isinstance(threshold, (int, float)):
            raise OperatorError("EMBEDDING_DISTANCE threshold must be a number")
        if threshold < 0:
            raise OperatorError("EMBEDDING_DISTANCE threshold must be non-negative")
            
        vec1 = self.encoder.encode(unit.content, unit.language)
        vec2 = self.encoder.encode(target, unit.language)
        distance = self.encoder.compute_distance(vec1, vec2)
        
        return distance <= threshold

class VectorSimilarityOperator(BaseOperator):
    """Vector similarity operator using a configurable encoder."""
    
    def __init__(self, encoder: Optional[BaseVectorEncoder] = None):
        """
        Initialize the similarity operator.
        
        Args:
            encoder (BaseVectorEncoder, optional): Custom vector encoder.
                If None, uses the SimpleVectorEncoder.
        """
        super().__init__("vector_similar")
        self.encoder = encoder or SimpleVectorEncoder()

    def __call__(
        self,
        unit: TextUnit,
        target: str,
        threshold: float = 0.5
    ) -> bool:
        """
        Check if the vector similarity is above the threshold.
        
        Args:
            unit (TextUnit): Text unit to analyze
            target (str): Text to compare with
            threshold (float): Minimum similarity score (0 to 1)
            
        Returns:
            bool: Whether the similarity is above threshold
            
        Raises:
            OperatorError: If arguments are invalid
        """
        if not isinstance(target, str):
            raise OperatorError("VECTOR_SIMILAR requires a string argument")
        if not isinstance(threshold, (int, float)):
            raise OperatorError("VECTOR_SIMILAR threshold must be a number")
        if not 0 <= threshold <= 1:
            raise OperatorError("VECTOR_SIMILAR threshold must be between 0 and 1")
            
        vec1 = self.encoder.encode(unit.content, unit.language)
        vec2 = self.encoder.encode(target, unit.language)
        similarity = self.encoder.compute_similarity(vec1, vec2)
        
        return similarity >= threshold

class SentimentOperator(BaseOperator):
    """Sentiment analysis operator that uses a configurable analyzer."""
    
    def __init__(self, analyzer=None):
        """
        Initialize the sentiment operator.
        
        Args:
            analyzer (BaseSentimentAnalyzer, optional): Custom sentiment analyzer.
                If None, uses the default WordlistSentimentAnalyzer.
        """
        super().__init__("sentiment")
        from .sentiment import WordlistSentimentAnalyzer
        self.analyzer = analyzer or WordlistSentimentAnalyzer()

    def __call__(self, unit: TextUnit, sentiment: str) -> bool:
        """
        Check if the text unit has the specified sentiment.
        
        Args:
            unit (TextUnit): Text unit to analyze
            sentiment (str): Expected sentiment ('positive', 'negative', 'neutral')
            
        Returns:
            bool: Whether the text has the specified sentiment
            
        Raises:
            OperatorError: If sentiment value is invalid
        """
        if sentiment not in ('positive', 'negative', 'neutral'):
            raise OperatorError(
                "SENTIMENT_IS requires 'positive', 'negative', or 'neutral'"
            )
        
        detected_sentiment = self.analyzer.analyze(
            unit.content,
            unit.language
        )
        return detected_sentiment == sentiment

class OperatorFactory:
    """Factory class for creating operators."""
    
    def __init__(self):
        """Initialize the operator factory."""
        self._operators = {}
        self._register_default_operators()

    def _register_default_operators(self):
        """Register the default set of operators."""
        # Text matching operators
        self._operators[TokenType.CONTAINS] = ContainsOperator
        self._operators[TokenType.STARTS_WITH] = StartsWithOperator
        self._operators[TokenType.ENDS_WITH] = EndsWithOperator
        
        # Length comparison operators
        self._operators[TokenType.LT] = lambda: LengthOperator(ComparisonType.LESS_THAN)
        self._operators[TokenType.GT] = lambda: LengthOperator(ComparisonType.GREATER_THAN)
        self._operators[TokenType.LE] = lambda: LengthOperator(ComparisonType.LESS_EQUAL)
        self._operators[TokenType.GE] = lambda: LengthOperator(ComparisonType.GREATER_EQUAL)
        self._operators[TokenType.EQ] = lambda: LengthOperator(ComparisonType.EQUALS)
        
        # Semantic operators
        self._operators[TokenType.TOPIC_IS] = TopicOperator
        self._operators[TokenType.SENTIMENT_IS] = SentimentOperator
        self._operators[TokenType.SIMILAR_TO] = SimilarToOperator
        self._operators[TokenType.VECTOR_SIMILAR] = VectorSimilarityOperator
        self._operators[TokenType.EMBEDDING_DISTANCE] = EmbeddingDistanceOperator   
        

    def register_operator(
        self, 
        token_type: TokenType, 
        operator: Union[BaseOperator, Type[BaseOperator], Callable[[], BaseOperator]]
    ):
        """
        Register a new operator.
        
        Args:
            token_type (TokenType): Token type for the operator
            operator: Either a BaseOperator instance, BaseOperator class, or factory function
            
        Raises:
            ValueError: If operator is not valid
        """
        if isinstance(operator, BaseOperator):
            self._operators[token_type] = lambda: operator
        elif isinstance(operator, type) and issubclass(operator, BaseOperator):
            self._operators[token_type] = operator
        elif callable(operator):
            # Test the factory function
            test_operator = operator()
            if not isinstance(test_operator, BaseOperator):
                raise ValueError("Factory function must return a BaseOperator instance")
            self._operators[token_type] = operator
        else:
            raise ValueError("Operator must be a BaseOperator instance, class, or factory function")

    def create_operator(self, token_type: TokenType) -> BaseOperator:
        """
        Create an operator instance.
        
        Args:
            token_type (TokenType): Token type of the operator
            
        Returns:
            BaseOperator: Instance of the requested operator
            
        Raises:
            OperatorError: If operator type is not supported
        """
        operator_class = self._operators.get(token_type)
        if not operator_class:
            raise OperatorError(f"Unsupported operator type: {token_type}")
            
        if isinstance(operator_class, type):
            return operator_class()
        else:
            return operator_class()  # For lambda defined operators