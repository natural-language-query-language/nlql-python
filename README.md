# NLQL (Natural Language Query Language)

> A SQL-like query language designed specifically for natural language processing and text retrieval.

## Overview

NLQL is a query language that brings the power and simplicity of SQL to natural language processing. It provides a structured way to query and analyze unstructured text data, with support for multilingual text processing, semantic analysis, and extensible operators.

## Key Features

- SQL-like syntax for intuitive querying
- Multiple text unit support (character, word, sentence, paragraph, document)
- Multilingual support (English, Chinese, Japanese, and mixed text)
- Rich set of operators for text analysis
- Pluggable semantic analysis components
- Built-in caching and optimization
- Extensible operator system

## Basic Syntax

```sql
SELECT <UNIT> 
[FROM <SOURCE>]
[WHERE <CONDITIONS>]
[GROUP BY <FIELD>]
[ORDER BY <FIELD>]
[LIMIT <NUMBER>]
```

### Query Units
- `CHAR`: Character level
- `WORD`: Word level
- `SENTENCE`: Sentence level
- `PARAGRAPH`: Paragraph level
- `DOCUMENT`: Document level

### Basic Operators
```sql
CONTAINS("text")              -- Contains specified text
STARTS_WITH("text")          -- Starts with specified text
ENDS_WITH("text")            -- Ends with specified text
LENGTH(<|>|=|<=|>=) number   -- Length conditions
```

### Semantic Operators
```sql
SIMILAR_TO("text", threshold)     -- Semantic similarity
TOPIC_IS("topic")                 -- Topic matching
SENTIMENT_IS("positive"|"negative"|"neutral")  -- Sentiment analysis
```

### Vector Operators
```sql
EMBEDDING_DISTANCE("text", threshold)  -- Vector distance
VECTOR_SIMILAR("vector", threshold)    -- Vector similarity
```

## Usage Examples

### Basic Usage
```python
from nlql import NLQL

# Initialize NLQL
nlql = NLQL()

# Add text for querying
nlql.text("Sample text for analysis...")

# Execute query
results = nlql.execute("SELECT SENTENCE WHERE CONTAINS('example')")

# Print results
for result in results:
    print(result.content)
```

### Custom Semantic Analysis
```python
from nlql import NLQLBuilder, Language
from nlql.executor.sentiment import BaseSentimentAnalyzer
from nlql.executor.operators import SentimentOperator

# Create custom sentiment analyzer
class MyCustomAnalyzer(BaseSentimentAnalyzer):
    def analyze(self, text: str, language: Language) -> str:
        # Your custom implementation
        return "positive"  # or "negative"/"neutral"

# Initialize NLQL with custom analyzer
nlql = (NLQLBuilder()
    .with_operator('SENTIMENT_IS', SentimentOperator(MyCustomAnalyzer()))
    .build())

# Use in queries
results = nlql.execute("SELECT SENTENCE WHERE SENTIMENT_IS('positive')")
```

### Custom Semantic Matching
```python
from nlql.executor.semantic import BaseSemanticMatcher
from nlql.executor.semantic_operators import SimilarToOperator

class MySemanticMatcher(BaseSemanticMatcher):
    def compute_similarity(self, text1: str, text2: str, language: Language) -> float:
        # Your custom implementation (e.g., using embeddings)
        return my_similarity_function(text1, text2)

nlql = (NLQLBuilder()
    .with_operator('SIMILAR_TO', SimilarToOperator(MySemanticMatcher()))
    .build())

results = nlql.execute("SELECT SENTENCE WHERE SIMILAR_TO('reference text', 0.8)")
```

### Custom Vector Operations
```python
from nlql.executor.semantic import BaseVectorEncoder
from nlql.executor.semantic_operators import EmbeddingDistanceOperator

class MyVectorEncoder(BaseVectorEncoder):
    def encode(self, text: str, language: Language) -> np.ndarray:
        # Your custom implementation (e.g., using BERT)
        return my_embedding_model.encode(text)

nlql = (NLQLBuilder()
    .with_operator('EMBEDDING_DISTANCE', 
                  EmbeddingDistanceOperator(MyVectorEncoder()))
    .build())

results = nlql.execute(
    "SELECT SENTENCE WHERE EMBEDDING_DISTANCE('reference', 0.5)"
)
```

### Adding Metadata Extractors
```python
nlql = NLQL()

# Register metadata extractors
nlql.register_metadata_extractor('word_count', lambda x: len(x.split()))
nlql.register_metadata_extractor('length', len)

# Use in queries
results = nlql.execute("""
    SELECT SENTENCE 
    WHERE LENGTH > 10 
    ORDER BY word_count 
    LIMIT 5
""")
```

### Performance Optimization
```python
from nlql import NLQLConfig

# Configure optimization features
config = NLQLConfig(
    use_cache=True,
    cache_capacity=1000,
    cache_ttl=3600,  # 1 hour
    use_index=True,
    enable_statistics=True
)

# Create optimized instance
nlql = NLQL(config)

# Check performance statistics
stats = nlql.get_statistics()
print(f"Cache hit ratio: {stats['cache_hit_ratio']}")
```

## Extension System

NLQL provides several base classes for extension:

- `BaseSentimentAnalyzer`: For custom sentiment analysis
- `BaseTopicAnalyzer`: For custom topic matching
- `BaseSemanticMatcher`: For custom semantic similarity
- `BaseVectorEncoder`: For custom vector operations
- `BaseOperator`: For entirely new operators

## Installation

```bash
pip install nlql
```

## Contributing

We welcome contributions! Please feel free to submit pull requests.

## License

This project is licensed under the MIT License.