from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

class TokenType(Enum):
    # Keywords
    SELECT = auto()
    FROM = auto()
    WHERE = auto()
    GROUP = auto()
    ORDER = auto()
    BY = auto()
    LIMIT = auto()
    
    # Units
    CHAR = auto()
    WORD = auto()
    SENTENCE = auto()
    PARAGRAPH = auto()
    DOCUMENT = auto()
    
    # Operators
    CONTAINS = auto()
    STARTS_WITH = auto()
    ENDS_WITH = auto()
    SIMILAR_TO = auto()
    TOPIC_IS = auto()
    SENTIMENT_IS = auto()
    VECTOR_SIMILAR = auto()
    EMBEDDING_DISTANCE = auto()
    
    # Comparison operators
    LT = auto()  # <
    GT = auto()  # >
    EQ = auto()  # =
    LE = auto()  # <=
    GE = auto()  # >=
    NE = auto()  # !=
    
    # Logical operators
    AND = auto()
    OR = auto()
    NOT = auto()
    
    # Delimiters
    LPAREN = auto()  # (
    RPAREN = auto()  # )
    COMMA = auto()   # ,
    DOT = auto()     # .
    
    # Other
    STRING = auto()    # String literal
    NUMBER = auto()    # Numeric literal
    IDENTIFIER = auto() # Identifier
    EOF = auto()       # End of file

@dataclass
class Token:
    """Represents a token in the query language."""
    type: TokenType
    value: Any
    position: int
    line: int
    column: int

    def __str__(self):
        return f"Token({self.type}, '{self.value}', pos={self.position}, line={self.line}, col={self.column})"

# Keyword mapping
KEYWORDS = {
    'SELECT': TokenType.SELECT,
    'FROM': TokenType.FROM,
    'WHERE': TokenType.WHERE,
    'GROUP': TokenType.GROUP,
    'ORDER': TokenType.ORDER,
    'BY': TokenType.BY,
    'LIMIT': TokenType.LIMIT,
    'AND': TokenType.AND,
    'OR': TokenType.OR,
    'NOT': TokenType.NOT,
    
    # Units
    'CHAR': TokenType.CHAR,
    'WORD': TokenType.WORD,
    'SENTENCE': TokenType.SENTENCE,
    'PARAGRAPH': TokenType.PARAGRAPH,
    'DOCUMENT': TokenType.DOCUMENT,
    
    # Operators
    'CONTAINS': TokenType.CONTAINS,
    'STARTS_WITH': TokenType.STARTS_WITH,
    'ENDS_WITH': TokenType.ENDS_WITH,
    'SIMILAR_TO': TokenType.SIMILAR_TO,
    'TOPIC_IS': TokenType.TOPIC_IS,
    'SENTIMENT_IS': TokenType.SENTIMENT_IS,
}