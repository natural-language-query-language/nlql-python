from typing import Any
from enum import Enum
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

try:
    import jieba
    JIEBA_AVAILABLE = True
    
    class TopicOperator(BaseOperator):
        """Operator that checks text topic using jieba."""
        
        def __init__(self):
            """Initialize the topic operator."""
            super().__init__("topic")
            jieba.initialize()
            
        def __call__(self, unit: TextUnit, topic: str) -> bool:
            """Check if the text unit belongs to the specified topic."""
            if not isinstance(topic, str):
                raise OperatorError("TOPIC_IS requires a string argument")
                
            # Get keywords using jieba
            keywords = jieba.analyse.extract_tags(unit.content, topK=5)
            topic_keywords = jieba.analyse.extract_tags(topic, topK=5)
            
            # Check keyword overlap
            overlap = set(keywords) & set(topic_keywords)
            return len(overlap) > 0
            
except ImportError:
    class TopicOperator(BaseOperator):
        """Dummy topic operator when jieba is not available."""
        
        def __call__(self, unit: TextUnit, topic: str) -> bool:
            """Always return False when jieba is not available."""
            raise OperatorError("TOPIC_IS operator requires jieba library")

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

    def register_operator(self, token_type: TokenType, operator_class: type):
        """
        Register a new operator.
        
        Args:
            token_type (TokenType): Token type for the operator
            operator_class (type): Operator class to register
        """
        if not issubclass(operator_class, BaseOperator):
            raise ValueError("Operator class must inherit from BaseOperator")
        self._operators[token_type] = operator_class

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