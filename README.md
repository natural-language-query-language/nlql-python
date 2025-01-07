# NLQL (Natural Language Query Language)

> A SQL-like query language designed specifically for natural language processing and text retrieval.

## Overview

NLQL is a query language that brings the power and simplicity of SQL to natural language processing. It provides a structured way to query and analyze unstructured text data, making it particularly useful for RAG (Retrieval-Augmented Generation) systems and large language models.

## Key Features

- SQL-like syntax for intuitive querying
- Multiple text unit support (character, word, sentence, paragraph, document)
- Rich set of operators for text analysis
- Semantic search capabilities
- Vector embedding support
- Extensible plugin system
- Performance optimizations with indexing and caching

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

### Basic Queries
```sql
-- Find sentences containing "artificial intelligence"
SELECT SENTENCE WHERE CONTAINS("artificial intelligence")

-- Find paragraphs with less than 100 characters
SELECT PARAGRAPH WHERE LENGTH < 100
```

### Advanced Queries
```sql
-- Find semantically similar sentences
SELECT SENTENCE 
WHERE SIMILAR_TO("How to improve productivity", 0.8)

-- Find positive sentences about innovation
SELECT SENTENCE 
WHERE CONTAINS("innovation") 
AND SENTIMENT_IS("positive")
ORDER BY LENGTH
LIMIT 10
```

## Implementation

The system is implemented with three main components:

1. **Tokenizer**: Breaks down query strings into tokens
2. **Parser**: Converts tokens into an abstract syntax tree (AST)
3. **Executor**: Executes the query and returns results

### Performance Optimizations

- Inverted index for text search
- Vector index for semantic search
- Query result caching
- Parallel processing for large datasets

## Extension System

NLQL supports custom extensions through:

1. Plugin System
   - Register custom operators
   - Add new query units
   - Implement custom functions

## Getting Started

1. Install the package:
```bash
pip install nlql
```

2. Basic usage:
```python
from nlql import NLQL

# Initialize NLQL
nlql = NLQL()

# Add text for querying
raw_text = """
Natural Language Processing (NLP) is a branch of artificial intelligence 
that helps computers understand human language. This technology is used 
in many applications. For example, virtual assistants use NLP to 
understand your commands.
"""
nlql.text(raw_text)

# Execute query
results = nlql.execute("SELECT SENTENCE WHERE CONTAINS('artificial intelligence')")

# Print results
for result in results:
    print(result)
```

## Contributing

We welcome contributions! Please see our contributing guidelines for more details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.