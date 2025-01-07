import unittest
from nlql.lexer.lexer import Lexer
from nlql.lexer.token import TokenType
from nlql.parser.parser import Parser
from nlql.parser.ast import (
    Query,
    BinaryCondition,
    UnaryCondition,
    OperatorCondition,
    ComparisonCondition
)
from nlql.exceptions import ParserError

class TestParser(unittest.TestCase):
    """Test suite for the parser."""

    def parse_query(self, query_text: str) -> Query:
        """Helper method to parse a query string."""
        lexer = Lexer(query_text)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        return parser.parse()

    def test_basic_select(self):
        """Test parsing of basic SELECT statement."""
        ast = self.parse_query("SELECT SENTENCE")
        
        self.assertEqual(ast.unit, TokenType.SENTENCE)
        self.assertIsNone(ast.source)
        self.assertIsNone(ast.conditions)
        self.assertIsNone(ast.group_by)
        self.assertIsNone(ast.order_by)
        self.assertIsNone(ast.limit)

    def test_select_with_source(self):
        """Test parsing SELECT with FROM clause."""
        ast = self.parse_query('SELECT PARAGRAPH FROM "test.txt"')
        
        self.assertEqual(ast.unit, TokenType.PARAGRAPH)
        self.assertEqual(ast.source, "test.txt")

    def test_select_with_simple_condition(self):
        """Test parsing SELECT with simple WHERE condition."""
        ast = self.parse_query(
            'SELECT SENTENCE WHERE CONTAINS("test")'
        )
        
        self.assertEqual(ast.unit, TokenType.SENTENCE)
        self.assertIsInstance(ast.conditions, OperatorCondition)
        self.assertEqual(ast.conditions.operator, TokenType.CONTAINS)
        self.assertEqual(ast.conditions.arguments, ["test"])

    def test_select_with_complex_conditions(self):
        """Test parsing SELECT with complex WHERE conditions."""
        ast = self.parse_query(
            'SELECT SENTENCE WHERE CONTAINS("test") AND LENGTH > 10'
        )
        
        self.assertIsInstance(ast.conditions, BinaryCondition)
        self.assertEqual(ast.conditions.operator, TokenType.AND)
        self.assertIsInstance(ast.conditions.left, OperatorCondition)
        self.assertIsInstance(ast.conditions.right, ComparisonCondition)

    def test_select_with_not_condition(self):
        """Test parsing SELECT with NOT condition."""
        ast = self.parse_query(
            'SELECT SENTENCE WHERE NOT CONTAINS("test")'
        )
        
        self.assertIsInstance(ast.conditions, UnaryCondition)
        self.assertEqual(ast.conditions.operator, TokenType.NOT)
        self.assertIsInstance(ast.conditions.operand, OperatorCondition)

    def test_select_with_all_clauses(self):
        """Test parsing SELECT with all optional clauses."""
        ast = self.parse_query(
            'SELECT SENTENCE FROM "test.txt" '
            'WHERE CONTAINS("test") '
            'GROUP BY topic '
            'ORDER BY length '
            'LIMIT 10'
        )
        
        self.assertEqual(ast.unit, TokenType.SENTENCE)
        self.assertEqual(ast.source, "test.txt")
        self.assertIsInstance(ast.conditions, OperatorCondition)
        self.assertEqual(ast.group_by, "topic")
        self.assertEqual(ast.order_by, "length")
        self.assertEqual(ast.limit, 10)

    def test_invalid_unit(self):
        """Test error handling for invalid unit type."""
        with self.assertRaises(ParserError):
            self.parse_query("SELECT INVALID")

    def test_invalid_condition(self):
        """Test error handling for invalid condition syntax."""
        with self.assertRaises(ParserError):
            self.parse_query("SELECT SENTENCE WHERE")

    def test_unclosed_parentheses(self):
        """Test error handling for unclosed parentheses."""
        with self.assertRaises(ParserError):
            self.parse_query(
                'SELECT SENTENCE WHERE (CONTAINS("test")'
            )

    def test_missing_operator_arguments(self):
        """Test error handling for missing operator arguments."""
        with self.assertRaises(ParserError):
            self.parse_query(
                'SELECT SENTENCE WHERE CONTAINS()'
            )

if __name__ == '__main__':
    unittest.main()