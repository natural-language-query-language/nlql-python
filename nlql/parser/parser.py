from typing import List
from ..lexer.token import Token, TokenType
from ..exceptions import ParserError
from .ast import (
    Query,
    Condition,
    BinaryCondition,
    UnaryCondition,
    OperatorCondition,
    ComparisonCondition
)

class Parser:
    """
    Parser for the text query language.
    
    Implements a recursive descent parser that converts a stream of tokens
    into an Abstract Syntax Tree (AST).
    """
    
    def __init__(self, tokens: List[Token]):
        """
        Initialize the parser with a list of tokens.
        
        Args:
            tokens (List[Token]): List of tokens from the lexer
        """
        self.tokens = tokens
        self.current_pos = 0
        self.current_token = tokens[0] if tokens else None

    def error(self, message: str) -> None:
        """
        Raise a parser error with the current token information.
        
        Args:
            message (str): Error message
            
        Raises:
            ParserError: Always raised with current token information
        """
        if self.current_token:
            raise ParserError(message, self.current_token)
        raise ParserError(message)

    def eat(self, token_type: TokenType) -> Token:
        """
        Consume a token of a specific type and advance to next token.
        
        Args:
            token_type (TokenType): Expected token type
            
        Returns:
            Token: The consumed token
            
        Raises:
            ParserError: If current token doesn't match expected type
        """
        if self.current_token.type == token_type:
            token = self.current_token
            self.advance()
            return token
        self.error(
            f"Expected {token_type.name}, got {self.current_token.type.name}"
        )

    def advance(self) -> None:
        """Advance to the next token in the token stream."""
        self.current_pos += 1
        if self.current_pos < len(self.tokens):
            self.current_token = self.tokens[self.current_pos]
        else:
            self.current_token = None

    def parse(self) -> Query:
        """
        Parse the complete query and return the AST.
        
        Returns:
            Query: The root node of the AST
            
        Raises:
            ParserError: If the query syntax is invalid
        """
        query = self.parse_select_statement()
        if self.current_token and self.current_token.type != TokenType.EOF:
            self.error("Expected end of input")
        return query

    def parse_select_statement(self) -> Query:
        """
        Parse a SELECT statement.
        
        Grammar:
        select_statement : SELECT unit [FROM source] 
                         [WHERE condition] 
                         [GROUP BY field] 
                         [ORDER BY field] 
                         [LIMIT number]
        
        Returns:
            Query: AST node representing the select statement
        """
        # Parse SELECT and unit
        self.eat(TokenType.SELECT)
        unit_token = self.current_token
        if not self._is_unit_type(unit_token.type):
            self.error(f"Expected a unit type, got {unit_token.type.name}")
        self.advance()

        # Initialize query components
        source = None
        conditions = None
        group_by = None
        order_by = None
        limit = None

        # Parse optional clauses
        while self.current_token and self.current_token.type != TokenType.EOF:
            if self.current_token.type == TokenType.FROM:
                self.advance()
                source = self.current_token.value
                self.advance()
                
            elif self.current_token.type == TokenType.WHERE:
                self.advance()
                if self.current_token and self.current_token.type != TokenType.EOF:
                    conditions = self.parse_conditions()
                else:
                    self.error("Expected condition after WHERE clause")
                
            elif self.current_token.type == TokenType.GROUP:
                self.eat(TokenType.GROUP)
                self.eat(TokenType.BY)
                group_by = self.current_token.value
                self.advance()
                
            elif self.current_token.type == TokenType.ORDER:
                self.eat(TokenType.ORDER)
                self.eat(TokenType.BY)
                order_by = self.current_token.value
                self.advance()
                
            elif self.current_token.type == TokenType.LIMIT:
                self.advance()
                limit_token = self.eat(TokenType.NUMBER)
                limit = limit_token.value
                
            else:
                self.error(f"Unexpected token: {self.current_token.type.name}")

        return Query(
            unit=unit_token.type,
            source=source,
            conditions=conditions,
            group_by=group_by,
            order_by=order_by,
            limit=limit
        )

    def parse_conditions(self) -> Condition:
        """
        Parse conditions in WHERE clause.
        
        Grammar:
        condition : condition AND condition
                 | condition OR condition
                 | NOT condition
                 | operator_condition
                 | comparison_condition
                 | (condition)
        
        Returns:
            Condition: AST node representing the condition
        """
        return self.parse_logical_or()

    def parse_logical_or(self) -> Condition:
        """Parse OR conditions."""
        condition = self.parse_logical_and()
        
        while (self.current_token and 
               self.current_token.type == TokenType.OR):
            operator = self.current_token
            self.advance()
            right = self.parse_logical_and()
            condition = BinaryCondition(
                operator=operator.type,
                left=condition,
                right=right
            )
            
        return condition

    def parse_logical_and(self) -> Condition:
        """Parse AND conditions."""
        condition = self.parse_primary_condition()
        
        while (self.current_token and 
               self.current_token.type == TokenType.AND):
            operator = self.current_token
            self.advance()
            right = self.parse_primary_condition()
            condition = BinaryCondition(
                operator=operator.type,
                left=condition,
                right=right
            )
            
        return condition

    def parse_primary_condition(self) -> Condition:
        """Parse primary conditions (NOT, parentheses, operator, comparison)."""
        if not self.current_token:
            self.error("Unexpected end of input in condition")
            
        # Early validation of condition start
        if self.current_token.type == TokenType.EOF:
            self.error("Expected condition after WHERE clause")

        # Handle NOT
        if self.current_token.type == TokenType.NOT:
            self.advance()
            condition = self.parse_primary_condition()
            return UnaryCondition(
                operator=TokenType.NOT,
                operand=condition
            )

        # Handle parentheses
        if self.current_token.type == TokenType.LPAREN:
            self.advance()
            condition = self.parse_conditions()
            self.eat(TokenType.RPAREN)
            return condition

        # Handle operator conditions
        if self._is_operator_type(self.current_token.type):
            return self.parse_operator_condition()

        # Handle comparison conditions
        return self.parse_comparison_condition()

    def parse_operator_condition(self) -> OperatorCondition:
        """Parse operator conditions (CONTAINS, STARTS_WITH, etc.)."""
        operator = self.current_token
        self.advance()
        
        arguments = []
        self.eat(TokenType.LPAREN)
        
        # Parse arguments
        while True:
            if self.current_token.type in (TokenType.STRING, TokenType.NUMBER):
                arguments.append(self.current_token.value)
                self.advance()
            else:
                self.error("Expected string or number argument")
                
            if self.current_token.type == TokenType.RPAREN:
                break
            self.eat(TokenType.COMMA)
            
        self.eat(TokenType.RPAREN)
        
        return OperatorCondition(
            operator=operator.type,
            arguments=arguments
        )

    def parse_comparison_condition(self) -> ComparisonCondition:
        """Parse comparison conditions (e.g., LENGTH > 10)."""
        if not self.current_token:
            self.error("Expected comparison condition")
            
        left = self.current_token.value
        self.advance()
        
        if not self.current_token:
            self.error("Incomplete comparison condition")
        
        if not self._is_comparison_operator(self.current_token.type):
            self.error("Expected comparison operator")
            
        operator = self.current_token
        self.advance()
        
        if self.current_token.type not in (TokenType.NUMBER, TokenType.STRING):
            self.error("Expected number or string value")
            
        right = self.current_token.value
        self.advance()
        
        return ComparisonCondition(
            operator=operator.type,
            left=left,
            right=right
        )

    def _is_unit_type(self, token_type: TokenType) -> bool:
        """Check if token type is a valid unit type."""
        return token_type in (
            TokenType.CHAR,
            TokenType.WORD,
            TokenType.SENTENCE,
            TokenType.PARAGRAPH,
            TokenType.DOCUMENT
        )

    def _is_operator_type(self, token_type: TokenType) -> bool:
        """Check if token type is a valid operator type."""
        return token_type in (
            TokenType.CONTAINS,
            TokenType.STARTS_WITH,
            TokenType.ENDS_WITH,
            TokenType.SIMILAR_TO,
            TokenType.TOPIC_IS,
            TokenType.SENTIMENT_IS
        )

    def _is_comparison_operator(self, token_type: TokenType) -> bool:
        """Check if token type is a valid comparison operator."""
        return token_type in (
            TokenType.LT,
            TokenType.GT,
            TokenType.EQ,
            TokenType.LE,
            TokenType.GE,
            TokenType.NE
        )