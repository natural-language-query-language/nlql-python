import unittest
from nlql import NLQL

def test_iterator_length(iterator):
    """
    Test the length of an iterator by converting it to a list.

    Args:
        iterator (Iterator): The iterator to test.

    Returns:
        int: The length of the iterator.
    """
    # Convert the iterator to a list
    iterator_list = list(iterator)

    # Calculate the length
    length = len(iterator_list)

    # Return the length
    return length


class TestNLQL(unittest.TestCase):
    """Test suite for the NLQL class."""

    def setUp(self):
        """Set up test fixtures."""
        self.nlql = NLQL()
        self.nlql.text(
            """This is the first paragraph. It contains multiple sentences! """)

    def test_basic_query(self):
        """Test basic query without conditions."""
        query = "SELECT SENTENCE"
        result = self.nlql.execute(query)
        self.assertEqual(test_iterator_length(result), 2)
        
    def test_contains_condition(self):
        """Test query with CONTAINS condition."""
        query = """SELECT SENTENCE WHERE CONTAINS('paragraph')"""
        result = self.nlql.execute(query)
        self.assertEqual(test_iterator_length(result), 1)
        
    def test_complex_condition(self):
        """Test query with complex conditions (AND, OR, NOT)."""
        query = """SELECT SENTENCE WHERE CONTAINS('paragraph') AND NOT CONTAINS("first")"""
        result = self.nlql.execute(query)
        self.assertEqual(test_iterator_length(result), 0)


if __name__ == '__main__':
    unittest.main()
