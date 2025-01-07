from typing import Dict, Any, Iterator, Optional, Type
from dataclasses import dataclass

from .lexer.lexer import Lexer
from .parser.parser import Parser
from .executor.executor import QueryExecutor
from .executor.operators import OperatorFactory, BaseOperator
from .executor.sentiment import BaseSentimentAnalyzer
from .utils.text_unit import TextUnit

@dataclass
class NLQLConfig:
    """Configuration options for NLQL."""
    use_cache: bool = True
    use_index: bool = True
    cache_capacity: int = 1000
    cache_ttl: int = 3600  # 1 hour
    enable_statistics: bool = True

class NLQL:
    """
    Natural Language Query Language main interface.
    
    This class provides a high-level interface for querying text data using
    a SQL-like query language, with support for optimization features.
    """
    
    def __init__(self, config: NLQLConfig = None):
        """
        Initialize NLQL with optional configuration.
        
        Args:
            config (NLQLConfig, optional): Configuration options
        """
        self.config = config or NLQLConfig()
        self._text_data = ""
        self._operator_factory = OperatorFactory()
        self._executor = QueryExecutor(
            self._operator_factory,
            use_cache=self.config.use_cache,
            use_index=self.config.use_index
        )
        self._metadata_extractors = {}

    def text(self, text: str) -> 'NLQL':
        """Set the text to be queried."""
        self._text_data = text
        return self

    def add_text(self, text: str) -> 'NLQL':
        """
        Append text to the existing text data.
        
        Args:
            text (str): Additional text content
            
        Returns:
            NLQL: Self for method chaining
        """
        self._text_data += "\n\n" + text
        # Clear cache since text has changed
        if self.config.use_cache:
            self._executor.cache.clear()
        return self

    def register_metadata_extractor(
        self,
        name: str,
        extractor: callable
    ) -> 'NLQL':
        """Register a metadata extractor function."""
        self._metadata_extractors[name] = extractor
        return self

    def register_operator(
        self,
        name: str,
        operator_class: Type[BaseOperator]
    ) -> 'NLQL':
        """Register a custom operator."""
        from .lexer.token import TokenType
        if not hasattr(TokenType, name):
            raise ValueError(f"Invalid operator name: {name}")
        token_type = getattr(TokenType, name)
        self._operator_factory.register_operator(token_type, operator_class)
        return self

    def set_sentiment_analyzer(
        self,
        analyzer: BaseSentimentAnalyzer
    ) -> 'NLQL':
        """Set a custom sentiment analyzer."""
        from .executor.operators import SentimentOperator
        from .lexer.token import TokenType
        sentiment_op = SentimentOperator(analyzer=analyzer)
        self._operator_factory.register_operator(
            TokenType.SENTIMENT_IS,
            lambda: sentiment_op
        )
        return self

    def clear_cache(self) -> 'NLQL':
        """
        Clear the query cache.
        
        Returns:
            NLQL: Self for method chaining
        """
        if self.config.use_cache:
            self._executor.cache.clear()
        return self

    def get_statistics(self) -> Optional[Dict[str, Any]]:
        """
        Get query execution statistics.
        
        Returns:
            Optional[Dict[str, Any]]: Statistics if enabled, None otherwise
        """
        if not self.config.enable_statistics:
            return None
            
        stats = self._executor.stats
        return {
            'total_queries': stats.total_queries,
            'cache_hits': stats.cache_hits,
            'cache_misses': stats.cache_misses,
            'cache_hit_ratio': stats.get_cache_hit_ratio(),
            'query_times': {
                query: stats.get_average_time(query)
                for query in stats.query_times
            }
        }

    def execute(self, query: str, text : None | str = None) -> Iterator[TextUnit]:
        """Execute a query on the current text."""
        if not self._text_data:
            raise ValueError("No text has been set. Use .text() first.")
            
        # Lexical analysis
        lexer = Lexer(query)
        tokens = lexer.tokenize()
        
        # Parsing
        parser = Parser(tokens)
        ast = parser.parse()
        
        # Execution
        return self._executor.execute(
            ast,
            text or self._text_data,
            self._metadata_extractors
        )

class NLQLBuilder:
    """Builder class for customizing NLQL instances."""
    
    def __init__(self):
        """Initialize the builder."""
        self._config = NLQLConfig()
        self._metadata_extractors = {}
        self._operators = {}
        self._sentiment_analyzer = None

    def with_cache(
        self,
        use_cache: bool = True,
        capacity: int = 1000,
        ttl: int = 3600
    ) -> 'NLQLBuilder':
        """Configure caching."""
        self._config.use_cache = use_cache
        self._config.cache_capacity = capacity
        self._config.cache_ttl = ttl
        return self

    def with_index(self, use_index: bool = True) -> 'NLQLBuilder':
        """Configure indexing."""
        self._config.use_index = use_index
        return self

    def with_statistics(self, enable: bool = True) -> 'NLQLBuilder':
        """Configure statistics collection."""
        self._config.enable_statistics = enable
        return self

    def with_metadata_extractor(
        self,
        name: str,
        extractor: callable
    ) -> 'NLQLBuilder':
        """Add a metadata extractor."""
        self._metadata_extractors[name] = extractor
        return self

    def with_operator(
        self,
        name: str,
        operator_class: Type[BaseOperator]
    ) -> 'NLQLBuilder':
        """Add a custom operator."""
        self._operators[name] = operator_class
        return self

    def with_sentiment_analyzer(
        self,
        analyzer: BaseSentimentAnalyzer
    ) -> 'NLQLBuilder':
        """Set a custom sentiment analyzer."""
        self._sentiment_analyzer = analyzer
        return self

    def build(self) -> NLQL:
        """Build and return customized NLQL instance."""
        nlql = NLQL(self._config)
        
        # Register metadata extractors
        for name, extractor in self._metadata_extractors.items():
            nlql.register_metadata_extractor(name, extractor)
            
        # Register operators
        for name, op_class in self._operators.items():
            nlql.register_operator(name, op_class)
            
        # Set sentiment analyzer if provided
        if self._sentiment_analyzer:
            nlql.set_sentiment_analyzer(self._sentiment_analyzer)
            
        return nlql