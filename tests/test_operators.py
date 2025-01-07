import unittest
from nlql.executor.sentiment import (
    BaseSentimentAnalyzer,
    WordlistSentimentAnalyzer
)
from nlql.utils.text_unit import Language
from nlql.executor.operators import SentimentOperator
from nlql.utils.text_unit import TextUnit
from nlql.lexer.token import TokenType

class CustomSentimentAnalyzer(BaseSentimentAnalyzer):
    """Example custom sentiment analyzer for testing."""
    
    def analyze(self, text: str, language: Language) -> str:
        """Always returns 'positive' for testing."""
        return 'positive'

class TestSentiment(unittest.TestCase):
    """Test suite for sentiment analysis."""

    def setUp(self):
        """Set up test fixtures."""
        self.text_unit_en = TextUnit(
            content="This is a great and wonderful product!",
            unit_type=TokenType.SENTENCE,
            start_pos=0,
            end_pos=35,
            metadata={},
            language=Language.ENGLISH
        )
        
        self.text_unit_zh = TextUnit(
            content="这个产品非常好，我很喜欢！",
            unit_type=TokenType.SENTENCE,
            start_pos=0,
            end_pos=14,
            metadata={},
            language=Language.CHINESE
        )

    def test_default_analyzer(self):
        """Test default wordlist analyzer."""
        analyzer = WordlistSentimentAnalyzer()
        
        # Test English text
        self.assertEqual(
            analyzer.analyze("This is great!", Language.ENGLISH),
            'positive'
        )
        self.assertEqual(
            analyzer.analyze("This is terrible!", Language.ENGLISH),
            'negative'
        )
        self.assertEqual(
            analyzer.analyze("This is normal.", Language.ENGLISH),
            'neutral'
        )
        
        # Test Chinese text
        self.assertEqual(
            analyzer.analyze("这个产品很好", Language.CHINESE),
            'positive'
        )
        self.assertEqual(
            analyzer.analyze("这个产品很差", Language.CHINESE),
            'negative'
        )

    def test_custom_analyzer(self):
        """Test custom analyzer integration."""
        custom_analyzer = CustomSentimentAnalyzer()
        operator = SentimentOperator(analyzer=custom_analyzer)
        
        # Should always return True for 'positive' due to custom analyzer
        self.assertTrue(operator(self.text_unit_en, 'positive'))
        self.assertTrue(operator(self.text_unit_zh, 'positive'))
        
        # Should always return False for 'negative' and 'neutral'
        self.assertFalse(operator(self.text_unit_en, 'negative'))
        self.assertFalse(operator(self.text_unit_en, 'neutral'))

    def test_custom_wordlist(self):
        """Test analyzer with custom word lists."""
        custom_positive = {'excellent', 'superb', '优质'}
        custom_negative = {'poor', 'bad', '差劲'}
        
        analyzer = WordlistSentimentAnalyzer(
            positive_words=custom_positive,
            negative_words=custom_negative
        )
        
        # Test with custom words
        self.assertEqual(
            analyzer.analyze("This is excellent and superb!", Language.ENGLISH),
            'positive'
        )
        self.assertEqual(
            analyzer.analyze("这个产品很优质", Language.CHINESE),
            'positive'
        )

    def test_mixed_language(self):
        """Test sentiment analysis with mixed language text."""
        analyzer = WordlistSentimentAnalyzer()
        text_unit_mixed = TextUnit(
            content="This product is great 非常好！",
            unit_type=TokenType.SENTENCE,
            start_pos=0,
            end_pos=25,
            metadata={},
            language=Language.MIXED
        )
        
        operator = SentimentOperator(analyzer=analyzer)
        self.assertTrue(operator(text_unit_mixed, 'positive'))

if __name__ == '__main__':
    unittest.main()