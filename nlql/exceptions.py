class TextQueryError(Exception):
    """Base exception class for text query language."""
    pass

class LexerError(TextQueryError):
    """Exception raised for errors during lexical analysis."""
    def __init__(self, message: str, position: int):
        self.position = position
        super().__init__(f"Lexer error at position {position}: {message}")

class ParserError(TextQueryError):
    """Exception raised for errors during parsing."""
    def __init__(self, message: str, token=None):
        self.token = token
        if token:
            message = f"Parser error at token '{token}': {message}"
        super().__init__(message)

class ExecutionError(TextQueryError):
    """Exception raised for errors during query execution."""
    pass

class ValidationError(TextQueryError):
    """Exception raised for query validation errors."""
    pass

class OperatorError(TextQueryError):
    """Exception raised for errors in operator execution."""
    pass

class UnsupportedError(TextQueryError):
    """Exception raised when attempting to use unsupported features."""
    pass