import unittest
from nlql.lexer.lexer import Lexer
from nlql.lexer.token import TokenType
from nlql.exceptions import LexerError

class TestLexer(unittest.TestCase):
    """Test suite for the lexical analyzer."""

    def test_basic_tokens(self):
        """Test basic token recognition."""
        query = "SELECT SENTENCE WHERE CONTAINS"
        lexer = Lexer(query)
        tokens = lexer.tokenize()
        
        expected_types = [
            TokenType.SELECT,
            TokenType.SENTENCE,
            TokenType.WHERE,
            TokenType.CONTAINS,
            TokenType.EOF
        ]
        
        self.assertEqual(len(tokens), len(expected_types))
        for token, expected_type in zip(tokens, expected_types):
            self.assertEqual(token.type, expected_type)

    def test_string_literal(self):
        """Test string literal recognition."""
        query = 'SELECT SENTENCE WHERE CONTAINS("hello world")'
        lexer = Lexer(query)
        tokens = lexer.tokenize()
        
        # Find the string token
        string_token = [t for t in tokens if t.type == TokenType.STRING][0]
        self.assertEqual(string_token.value, "hello world")

    def test_numbers(self):
        """Test numeric literal recognition."""
        query = "LIMIT 100"
        lexer = Lexer(query)
        tokens = lexer.tokenize()
        
        self.assertEqual(tokens[1].type, TokenType.NUMBER)
        self.assertEqual(tokens[1].value, 100)

    def test_operators(self):
        """Test operator recognition."""
        query = "LENGTH <= 100 AND LENGTH >= 50"
        lexer = Lexer(query)
        tokens = lexer.tokenize()
        
        operator_types = [
            TokenType.IDENTIFIER,  # LENGTH
            TokenType.LE,
            TokenType.NUMBER,
            TokenType.AND,
            TokenType.IDENTIFIER,  # LENGTH
            TokenType.GE,
            TokenType.NUMBER,
            TokenType.EOF
        ]
        
        self.assertEqual(len(tokens), len(operator_types))
        for token, expected_type in zip(tokens, operator_types):
            self.assertEqual(token.type, expected_type)

    def test_comments(self):
        """Test comment handling."""
        query = """
        -- This is a comment
        SELECT SENTENCE
        -- Another comment
        WHERE CONTAINS("text")
        """
        lexer = Lexer(query)
        tokens = lexer.tokenize()
        
        # Comments should be ignored, so we should only get the actual tokens
        expected_types = [
            TokenType.SELECT,
            TokenType.SENTENCE,
            TokenType.WHERE,
            TokenType.CONTAINS,
            TokenType.LPAREN,
            TokenType.STRING,
            TokenType.RPAREN,
            TokenType.EOF
        ]
        
        self.assertEqual(len(tokens), len(expected_types))
        for token, expected_type in zip(tokens, expected_types):
            self.assertEqual(token.type, expected_type)

    def test_invalid_character(self):
        """Test invalid character handling."""
        query = "SELECT SENTENCE WHERE @"
        lexer = Lexer(query)
        
        with self.assertRaises(LexerError):
            lexer.tokenize()

    def test_unterminated_string(self):
        """Test unterminated string literal handling."""
        query = 'SELECT SENTENCE WHERE CONTAINS("unterminated'
        lexer = Lexer(query)
        
        with self.assertRaises(LexerError):
            lexer.tokenize()

if __name__ == '__main__':
    unittest.main()