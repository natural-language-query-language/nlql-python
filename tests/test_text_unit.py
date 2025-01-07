import unittest
from nlql.utils.text_unit import TextUnitizer, Language
from nlql.lexer.token import TokenType

class TestTextUnitizer(unittest.TestCase):
    """Test suite for multilingual text unit processing."""

    def setUp(self):
        """Set up test fixtures."""
        self.unitizer = TextUnitizer(language=Language.MIXED)
        
        # Sample texts in different languages
        self.english_text = """This is English text. It has multiple sentences!
        
        This is a new paragraph."""
        
        self.chinese_text = """这是中文文本。这里有多个句子！
        
        这是新的段落。"""
        
        self.japanese_text = """これは日本語のテキストです。複数の文があります！
        
        新しい段落です。"""
        
        self.mixed_text = """This is mixed 中英文 text. 这是第二句。
        
        New paragraph 新段落。"""

    def test_language_detection(self):
        """Test language detection functionality."""
        test_cases = [
            ("This is English.", Language.ENGLISH),
            ("这是中文。", Language.CHINESE),
            ("これは日本語です。", Language.JAPANESE),
            ("This is 中英混合 text.", Language.MIXED),
            ("", Language.ENGLISH),  # Empty text defaults to English
            ("123 456", Language.ENGLISH),  # Numbers only
            ("Hello 世界！", Language.MIXED)
        ]
        
        for text, expected_language in test_cases:
            with self.subTest(text=text):
                detected = self.unitizer.detect_language(text)
                self.assertEqual(
                    detected, 
                    expected_language,
                    f"Failed to detect correct language for: {text}"
                )

    def test_char_splitting(self):
        """Test character splitting for different languages."""
        test_cases = [
            ("Hello", 5, Language.ENGLISH),
            ("你好", 2, Language.CHINESE),
            ("こんにちは", 5, Language.JAPANESE),
            ("Hello你好", 7, Language.MIXED)
        ]
        
        for text, expected_count, language in test_cases:
            with self.subTest(text=text):
                units = list(self.unitizer.split_into_units(
                    text,
                    TokenType.CHAR,
                    language=language
                ))
                self.assertEqual(len(units), expected_count)
                self.assertEqual("".join(u.content for u in units), text)

    def test_word_splitting(self):
        """Test word splitting for different languages."""
        # English word splitting
        english = "Hello world! Testing 123"
        eng_units = list(self.unitizer.split_into_units(
            english,
            TokenType.WORD,
            language=Language.ENGLISH
        ))
        self.assertEqual(
            [u.content for u in eng_units],
            ["Hello", "world", "Testing"]
        )

        # Chinese word splitting (depends on jieba)
        chinese = "我爱自然语言处理"
        chn_units = list(self.unitizer.split_into_units(
            chinese,
            TokenType.WORD,
            language=Language.CHINESE
        ))
        self.assertTrue(all(len(u.content) > 0 for u in chn_units))

        # Mixed language splitting
        mixed = "Hello 世界 World"
        mix_units = list(self.unitizer.split_into_units(
            mixed,
            TokenType.WORD,
            language=Language.MIXED
        ))
        self.assertTrue(any(u.content == "Hello" for u in mix_units))
        self.assertTrue(any(u.content == "World" for u in mix_units))
        self.assertTrue(any(len(u.content) > 0 for u in mix_units))

    def test_sentence_splitting(self):
        """Test sentence splitting for different languages."""
        test_cases = [
            (
                "This is one. This is two!",
                2,
                Language.ENGLISH
            ),
            (
                "这是第一句。这是第二句！",
                2,
                Language.CHINESE
            ),
            (
                "これは一つ目です。これは二つ目です！",
                2,
                Language.JAPANESE
            ),
            (
                "This is one. 这是第二句。",
                2,
                Language.MIXED
            )
        ]
        
        for text, expected_count, language in test_cases:
            with self.subTest(text=text):
                units = list(self.unitizer.split_into_units(
                    text,
                    TokenType.SENTENCE,
                    language=language
                ))
                self.assertEqual(len(units), expected_count)
                self.assertTrue(all(u.content.strip() for u in units))

    def test_paragraph_splitting(self):
        """Test paragraph splitting for all languages."""
        test_cases = [
            (self.english_text, 2),
            (self.chinese_text, 2),
            (self.japanese_text, 2),
            (self.mixed_text, 2)
        ]
        
        for text, expected_count in test_cases:
            with self.subTest(text=text[:20] + "..."):
                units = list(self.unitizer.split_into_units(
                    text,
                    TokenType.PARAGRAPH
                ))
                self.assertEqual(len(units), expected_count)
                self.assertTrue(all(u.content.strip() for u in units))

    def test_document_handling(self):
        """Test document-level processing."""
        test_cases = [
            (self.english_text, Language.ENGLISH),
            (self.chinese_text, Language.CHINESE),
            (self.japanese_text, Language.JAPANESE),
            (self.mixed_text, Language.MIXED)
        ]
        
        for text, language in test_cases:
            with self.subTest(language=language):
                units = list(self.unitizer.split_into_units(
                    text,
                    TokenType.DOCUMENT,
                    language=language
                ))
                self.assertEqual(len(units), 1)
                self.assertEqual(units[0].content, text)
                self.assertEqual(units[0].start_pos, 0)
                self.assertEqual(units[0].end_pos, len(text))

    def test_metadata_extraction(self):
        """Test metadata extraction functionality."""
        def word_count(text): return len(text.split())
        def char_count(text): return len(text)
        def has_punctuation(text): return any(p in text for p in ".!?。！？")
        
        extractors = {
            'word_count': word_count,
            'char_count': char_count,
            'has_punctuation': has_punctuation
        }
        
        test_cases = [
            ("Hello world.", 2, 12, True),
            ("你好世界。", 1, 5, True),
            ("No punctuation", 2, 14, False)
        ]
        
        for text, exp_words, exp_chars, exp_punct in test_cases:
            with self.subTest(text=text):
                units = list(self.unitizer.split_into_units(
                    text,
                    TokenType.SENTENCE,
                    metadata_extractors=extractors
                ))
                
                if units:  # Some texts might not form valid sentences
                    metadata = units[0].metadata
                    self.assertEqual(metadata['char_count'], exp_chars)
                    self.assertEqual(metadata['has_punctuation'], exp_punct)
                    # Word count might vary by language implementation
                    self.assertIsInstance(metadata['word_count'], int)

    def test_empty_and_whitespace(self):
        """Test handling of empty and whitespace-only inputs."""
        test_cases = [
            "",
            "   ",
            "\n\n",
            "\t\t",
            "\r\n"
        ]
        
        for text in test_cases:
            with self.subTest(text=repr(text)):
                for unit_type in [TokenType.CHAR, TokenType.WORD,
                                TokenType.SENTENCE, TokenType.PARAGRAPH]:
                    units = list(self.unitizer.split_into_units(text, unit_type))
                    self.assertEqual(len(units), 0)

    def test_position_tracking(self):
        """Test position tracking in text units."""
        text = "Hello 世界。"
        units = list(self.unitizer.split_into_units(
            text,
            TokenType.WORD,
            language=Language.MIXED
        ))
        
        for unit in units:
            # Verify that positions are valid
            self.assertTrue(0 <= unit.start_pos < len(text))
            self.assertTrue(0 < unit.end_pos <= len(text))
            self.assertTrue(unit.start_pos < unit.end_pos)
            
            # Verify content matches positions
            self.assertEqual(
                text[unit.start_pos:unit.end_pos].strip(),
                unit.content.strip()
            )

    def test_custom_patterns(self):
        """Test setting custom splitting patterns."""
        unitizer = TextUnitizer()
        
        # Set custom word pattern to include numbers
        unitizer.set_patterns(word_pattern=r'\b[\w\d]+\b')
        
        text = "Testing123 with 456 numbers"
        units = list(unitizer.split_into_units(text, TokenType.WORD))
        
        expected = ["Testing123", "with", "456", "numbers"]
        self.assertEqual([u.content for u in units], expected)

if __name__ == '__main__':
    unittest.main()