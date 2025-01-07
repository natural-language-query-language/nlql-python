import unittest
from nlql.executor.executor import QueryExecutor
from nlql.parser.ast import (
    Query, OperatorCondition, BinaryCondition,
    UnaryCondition, ComparisonCondition
)
from nlql.lexer.token import TokenType
from nlql.exceptions import ExecutionError

class TestQueryExecutor(unittest.TestCase):
    """Test suite for the query executor."""

    def setUp(self):
        """Set up test fixtures."""
        self.executor = QueryExecutor()
        
        # Sample text for testing
        self.text = """This is the first paragraph. It contains multiple sentences!
        
        This is the second paragraph. It's longer than the first one.
        It has more sentences. Some of them are very interesting.
        
        And this is the last paragraph."""

    def test_basic_query(self):
        """Test basic query without conditions."""
        query = Query(
            unit=TokenType.SENTENCE,
            source=None,
            conditions=None,
            group_by=None,
            order_by=None,
            limit=None
        )
        
        results = list(self.executor.execute(query, self.text))
        self.assertEqual(len(results), 7)  # Total number of sentences

    def test_contains_condition(self):
        """Test query with CONTAINS condition."""
        condition = OperatorCondition(
            operator=TokenType.CONTAINS,
            arguments=["paragraph"]
        )
        
        query = Query(
            unit=TokenType.SENTENCE,
            source=None,
            conditions=condition,
            group_by=None,
            order_by=None,
            limit=None
        )
        
        results = list(self.executor.execute(query, self.text))
        self.assertEqual(len(results), 3)  # Sentences containing "paragraph"

    def test_complex_condition(self):
        """Test query with complex conditions (AND, OR, NOT)."""
        # (contains 'paragraph' AND NOT contains 'last')
        condition = BinaryCondition(
            operator=TokenType.AND,
            left=OperatorCondition(
                operator=TokenType.CONTAINS,
                arguments=["paragraph"]
            ),
            right=UnaryCondition(
                operator=TokenType.NOT,
                operand=OperatorCondition(
                    operator=TokenType.CONTAINS,
                    arguments=["last"]
                )
            )
        )
        
        query = Query(
            unit=TokenType.SENTENCE,
            source=None,
            conditions=condition,
            group_by=None,
            order_by=None,
            limit=None
        )
        
        results = list(self.executor.execute(query, self.text))
        self.assertEqual(len(results), 2)  # First two paragraphs

    def test_metadata_and_ordering(self):
        """Test query with metadata extraction and ordering."""
        # Add length to metadata
        def get_length(text): return len(text)
        metadata_extractors = {'length': get_length}
        
        query = Query(
            unit=TokenType.SENTENCE,
            source=None,
            conditions=None,
            group_by=None,
            order_by='length',
            limit=None
        )
        
        results = list(self.executor.execute(
            query,
            self.text,
            metadata_extractors
        ))
        
        # Verify ordering
        lengths = [len(unit.content) for unit in results]
        self.assertEqual(lengths, sorted(lengths))

    def test_limit(self):
        """Test query with LIMIT clause."""
        query = Query(
            unit=TokenType.SENTENCE,
            source=None,
            conditions=None,
            group_by=None,
            order_by=None,
            limit=2
        )
        
        results = list(self.executor.execute(query, self.text))
        self.assertEqual(len(results), 2)

    def test_comparison_condition(self):
        """Test query with comparison condition."""
        # Add length to metadata
        def get_length(text): return len(text)
        metadata_extractors = {'length': get_length}
        
        condition = ComparisonCondition(
            operator=TokenType.GT,
            left='length',
            right=30  # Length greater than 30
        )
        
        query = Query(
            unit=TokenType.SENTENCE,
            source=None,
            conditions=condition,
            group_by=None,
            order_by=None,
            limit=None
        )
        
        results = list(self.executor.execute(
            query,
            self.text,
            metadata_extractors
        ))
        
        # Verify all results have length > 30
        self.assertTrue(all(len(unit.content) > 30 for unit in results))

    def test_error_handling(self):
        """Test error handling for invalid queries."""
        # Test invalid metadata field
        query = Query(
            unit=TokenType.SENTENCE,
            source=None,
            conditions=None,
            group_by='nonexistent_field',  # Invalid field
            order_by=None,
            limit=None
        )
        
        with self.assertRaises(ExecutionError):
            list(self.executor.execute(query, self.text))

    def test_empty_text(self):
        """Test handling of empty text."""
        query = Query(
            unit=TokenType.SENTENCE,
            source=None,
            conditions=None,
            group_by=None,
            order_by=None,
            limit=None
        )
        
        results = list(self.executor.execute(query, ""))
        self.assertEqual(len(results), 0)

if __name__ == '__main__':
    unittest.main()