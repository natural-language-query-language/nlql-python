from typing import List, Optional
from .token import Token, TokenType, KEYWORDS
from ..exceptions import LexerError


class Lexer:
    """
    A lexical analyzer for the text query language.

    This lexer converts an input string into a stream of tokens that can be
    consumed by the parser. It handles keywords, operators, literals, and
    identifiers according to the language specification.

    Attributes:
        text (str): The input text to be tokenized
        position (int): Current position in the input
        line (int): Current line number
        column (int): Current column number
        current_char (Optional[str]): Current character being processed
    """

    def __init__(self, text: str):
        """
        Initialize the lexer with input text.

        Args:
            text (str): The input text to be tokenized
        """
        self.text = text
        self.position = 0
        self.line = 1
        self.column = 1
        self.current_char = text[0] if text else None

    def error(self, message: str) -> None:
        """
        Raise a lexer error with position information.

        Args:
            message (str): Error message describing the issue

        Raises:
            LexerError: Always raised with position information
        """
        raise LexerError(message, self.position)

    def advance(self) -> None:
        """Move to the next character in the input text."""
        self.position += 1
        self.column += 1

        if self.position > len(self.text) - 1:
            self.current_char = None
        else:
            if self.current_char == '\n':
                self.line += 1
                self.column = 1
            self.current_char = self.text[self.position]

    def skip_whitespace(self) -> None:
        """Skip whitespace characters while keeping track of line numbers."""
        while self.current_char and self.current_char.isspace():
            self.advance()

    def skip_comment(self) -> None:
        """Skip single-line comments starting with '--'."""
        while self.current_char and self.current_char != '\n':
            self.advance()
        if self.current_char == '\n':
            self.advance()

    def read_string(self) -> Token:
        """
        Read a string literal enclosed in either double quotes or single quotes.

        Returns:
            Token: A STRING token containing the string value

        Raises:
            LexerError: If string is not properly terminated
        """
        start_pos = self.position
        start_line = self.line
        start_column = self.column

        quote_char = self.current_char
        self.advance()  # Skip opening quote
        value = ""

        while self.current_char and self.current_char != quote_char:
            if self.current_char == '\\':
                self.advance()
                if self.current_char == 'n':
                    value += '\n'
                elif self.current_char == 't':
                    value += '\t'
                elif self.current_char in ['"', "'", '\\']:
                    value += self.current_char
                else:
                    self.error(f"Invalid escape sequence: \\{
                               self.current_char}")
            else:
                value += self.current_char
            self.advance()

        if self.current_char != quote_char:
            self.error("Unterminated string literal")

        self.advance()  # Skip closing quote
        return Token(TokenType.STRING, value, start_pos, start_line, start_column)

    def read_number(self) -> Token:
        """
        Read a numeric literal (integer or float).

        Returns:
            Token: A NUMBER token containing the numeric value
        """
        start_pos = self.position
        start_line = self.line
        start_column = self.column

        value = ""
        is_float = False

        while self.current_char and (self.current_char.isdigit() or self.current_char == '.'):
            if self.current_char == '.':
                if is_float:
                    self.error(
                        "Invalid number format: multiple decimal points")
                is_float = True
            value += self.current_char
            self.advance()

        if is_float:
            return Token(TokenType.NUMBER, float(value), start_pos, start_line, start_column)
        return Token(TokenType.NUMBER, int(value), start_pos, start_line, start_column)

    def read_identifier(self) -> Token:
        """
        Read an identifier or keyword.

        Returns:
            Token: Either a keyword token or an IDENTIFIER token
        """
        start_pos = self.position
        start_line = self.line
        start_column = self.column

        value = ""
        while self.current_char and (self.current_char.isalnum() or self.current_char == '_'):
            value += self.current_char
            self.advance()

        token_type = KEYWORDS.get(value.upper(), TokenType.IDENTIFIER)
        return Token(token_type, value, start_pos, start_line, start_column)

    def get_next_token(self) -> Token:
        """
        Get the next token from the input text.
        
        Returns:
            Token: The next token in the input stream
            
        Raises:
            LexerError: If an invalid character is encountered
        """
        while self.current_char:
            # Skip whitespace
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
                
            # Skip comments
            if self.current_char == '-' and self.peek() == '-':
                self.advance()  # Skip first '-'
                self.advance()  # Skip second '-'
                self.skip_comment()
                continue

            # Current character position
            pos = self.position
            line = self.line
            col = self.column

            # String literals
            if self.current_char in ['"', "'"]:
                return self.read_string()

            # Numbers
            if self.current_char.isdigit():
                return self.read_number()

            # Identifiers and keywords
            if self.current_char.isalpha() or self.current_char == '_':
                return self.read_identifier()

            # Operators and delimiters
            if self.current_char == '<':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(TokenType.LE, '<=', pos, line, col)
                return Token(TokenType.LT, '<', pos, line, col)

            if self.current_char == '>':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(TokenType.GE, '>=', pos, line, col)
                return Token(TokenType.GT, '>', pos, line, col)

            if self.current_char == '=':
                self.advance()
                return Token(TokenType.EQ, '=', pos, line, col)

            if self.current_char == '(':
                self.advance()
                return Token(TokenType.LPAREN, '(', pos, line, col)

            if self.current_char == ')':
                self.advance()
                return Token(TokenType.RPAREN, ')', pos, line, col)

            if self.current_char == ',':
                self.advance()
                return Token(TokenType.COMMA, ',', pos, line, col)

            if self.current_char == '.':
                self.advance()
                return Token(TokenType.DOT, '.', pos, line, col)

            # Invalid character
            self.error(f"Invalid character: {self.current_char}")

        # End of input
        return Token(TokenType.EOF, None, self.position, self.line, self.column)

    def peek(self) -> Optional[str]:
        """
        Look at the next character without consuming it.

        Returns:
            Optional[str]: The next character or None if at end of input
        """
        peek_pos = self.position + 1
        if peek_pos > len(self.text) - 1:
            return None
        return self.text[peek_pos]

    def tokenize(self) -> List[Token]:
        """
        Tokenize the entire input text.

        Returns:
            List[Token]: A list of all tokens in the input

        Raises:
            LexerError: If any lexical error is encountered
        """
        tokens = []
        while True:
            token = self.get_next_token()
            tokens.append(token)
            if token.type == TokenType.EOF:
                break
        return tokens
