# utils/text_unit.py
from typing import List, Dict, Any, Iterator, Optional
import re
from dataclasses import dataclass
from enum import Enum, auto
from ..lexer.token import TokenType

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

class Language(Enum):
    """Supported languages."""
    ENGLISH = auto()
    CHINESE = auto()
    JAPANESE = auto()
    MIXED = auto()    # For mixed language content

@dataclass
class TextUnit:
    """
    Represents a unit of text with its content and metadata.
    
    Attributes:
        content (str): The actual text content
        unit_type (TokenType): Type of the unit (CHAR, WORD, etc.)
        start_pos (int): Starting position in the original text
        end_pos (int): Ending position in the original text
        metadata (Dict[str, Any]): Additional metadata about the unit
        language (Language): Language of this text unit
    """
    content: str
    unit_type: TokenType
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]
    language: Language

class TextUnitizer:
    """Handles the splitting of text into various units with multi-language support."""
    
    # Language-specific patterns
    PATTERNS = {
        Language.ENGLISH: {
            'sentence_end': r'[.!?][\'")\]}]*(?=\s|$)',
            'word': r'\b[a-zA-Z]+\b',
            'paragraph': r'\n\s*\n'
        },
        Language.CHINESE: {
            'sentence_end': r'([。！？])[」』\'\"]*(?!\w)',
            'word': None,
            'paragraph': r'\n\s*\n'
        },
        Language.JAPANESE: {
            'sentence_end': r'([。！？])[」』\'\"]*(?!\w)',
            'word': None,
            'paragraph': r'\n\s*\n'
        },
        Language.MIXED: {
            'sentence_end': r'(?:[.!?。！？][」』\'")\]}]*(?=\s|$|\S))',
            'word': r'\b[a-zA-Z]+\b',
            'paragraph': r'\n\s*\n'
        }
    }
    
    def __init__(self, language: Language = Language.MIXED):
        """Initialize the TextUnitizer for a specific language."""
        self.language = language
        self._load_patterns(language)
        
        if language in [Language.CHINESE, Language.MIXED] and JIEBA_AVAILABLE:
            jieba.initialize()

    def _load_patterns(self, language: Language) -> None:
        """Load the appropriate patterns for the specified language."""
        patterns = self.PATTERNS[language]
        
        self._sent_pattern = re.compile(patterns['sentence_end'])
        self._word_pattern = re.compile(patterns['word']) if patterns['word'] else None
        self._para_pattern = re.compile(patterns['paragraph'])

    def detect_language(self, text: str) -> Language:
        """Detect the primary language of a text."""
        text = text.strip()
        if not text:
            return Language.ENGLISH

        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total_chars = len(re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ffa-zA-Z]', text))
        if total_chars == 0:
            return Language.ENGLISH
            
        chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
        japanese_ratio = japanese_chars / total_chars if total_chars > 0 else 0
        english_ratio = english_chars / total_chars if total_chars > 0 else 0
        
        if english_ratio > 0.2 and (chinese_ratio > 0.1 or japanese_ratio > 0.1):
            return Language.MIXED
        elif chinese_ratio > 0.5:
            return Language.CHINESE
        elif japanese_ratio > 0.5:
            return Language.JAPANESE
        else:
            return Language.ENGLISH

    def _split_chars(
        self, 
        text: str, 
        metadata_extractors: Dict[str, callable],
        language: Language
    ) -> Iterator[TextUnit]:
        """Split text into individual characters."""
        text = text.strip()
        if not text:
            return

        for i, char in enumerate(text):
            if not char.strip():
                continue
            metadata = self._extract_metadata(char, metadata_extractors)
            yield TextUnit(
                content=char,
                unit_type=TokenType.CHAR,
                start_pos=i,
                end_pos=i + 1,
                metadata=metadata,
                language=language
            )

    def _split_words_chinese(self, text: str) -> Iterator[tuple[str, int, int]]:
        """Split Chinese text into words using jieba."""
        if not JIEBA_AVAILABLE:
            yield from self._split_words_by_char(text)
            return
            
        pos = 0
        for word in jieba.cut(text):
            if word.strip():
                start = text.find(word, pos)
                if start >= 0:
                    end = start + len(word)
                    pos = end
                    yield word, start, end

    def _split_words_by_char(self, text: str) -> Iterator[tuple[str, int, int]]:
        """Split text character by character (fallback for CJK languages)."""
        text = text.strip()
        for i, char in enumerate(text):
            if not char.isspace():
                yield char, i, i + 1

    def _split_words_mixed(self, text: str) -> Iterator[tuple[str, int, int]]:
        """Split text with mixed languages."""
        current_pos = 0
        text_length = len(text)
        
        while current_pos < text_length:
            char = text[current_pos]
            
            if char.isspace():
                current_pos += 1
                continue
                
            if '\u4e00' <= char <= '\u9fff' or '\u3040' <= char <= '\u30ff':
                start = current_pos
                while (current_pos < text_length and 
                       ('\u4e00' <= text[current_pos] <= '\u9fff' or 
                        '\u3040' <= text[current_pos] <= '\u30ff')):
                    current_pos += 1
                if JIEBA_AVAILABLE:
                    segment = text[start:current_pos]
                    for word in jieba.cut(segment):
                        if word.strip():
                            word_start = text.find(word, start)
                            yield word, word_start, word_start + len(word)
                else:
                    if text[start:current_pos].strip():
                        yield text[start:current_pos], start, current_pos
                    
            elif char.isalpha():
                start = current_pos
                while current_pos < text_length and text[current_pos].isalpha():
                    current_pos += 1
                if text[start:current_pos].strip():
                    yield text[start:current_pos], start, current_pos
            else:
                current_pos += 1

    def _split_sentences(
        self, 
        text: str, 
        metadata_extractors: Dict[str, callable],
        language: Language
    ) -> Iterator[TextUnit]:
        """Split text into sentences."""
        if not text.strip():
            return

        sentences = []
        current_pos = 0
        text = text.strip()
        
        while current_pos < len(text):
            match = self._sent_pattern.search(text, current_pos)
            if not match:
                # 处理剩余文本作为一个句子
                if text[current_pos:].strip():
                    sentences.append((text[current_pos:].strip(), current_pos, len(text)))
                break
            
            end_pos = match.end()
            sentence = text[current_pos:end_pos].strip()
            if sentence:
                sentences.append((sentence, current_pos, end_pos))
            current_pos = end_pos
        
        for sentence, start, end in sentences:
            metadata = self._extract_metadata(sentence, metadata_extractors)
            yield TextUnit(
                content=sentence,
                unit_type=TokenType.SENTENCE,
                start_pos=start,
                end_pos=end,
                metadata=metadata,
                language=language
            )

    def _split_paragraphs(
        self, 
        text: str, 
        metadata_extractors: Dict[str, callable],
        language: Language
    ) -> Iterator[TextUnit]:
        """Split text into paragraphs."""
        if not text.strip():
            return

        start = 0
        text = text.strip()
        
        for match in self._para_pattern.finditer(text):
            para = text[start:match.start()].strip()
            if para:
                metadata = self._extract_metadata(para, metadata_extractors)
                yield TextUnit(
                    content=para,
                    unit_type=TokenType.PARAGRAPH,
                    start_pos=start,
                    end_pos=match.start(),
                    metadata=metadata,
                    language=language
                )
            start = match.end()

        if start < len(text):
            para = text[start:].strip()
            if para:
                metadata = self._extract_metadata(para, metadata_extractors)
                yield TextUnit(
                    content=para,
                    unit_type=TokenType.PARAGRAPH,
                    start_pos=start,
                    end_pos=len(text),
                    metadata=metadata,
                    language=language
                )

    def split_into_units(
        self, 
        text: str, 
        unit_type: TokenType,
        metadata_extractors: Dict[str, callable] = None,
        language: Optional[Language] = None
    ) -> Iterator[TextUnit]:
        """Split text into specified units with language support."""
        if metadata_extractors is None:
            metadata_extractors = {}

        text_language = language or self.detect_language(text)

        if not text.strip():
            return

        if unit_type == TokenType.CHAR:
            yield from self._split_chars(text, metadata_extractors, text_language)
        elif unit_type == TokenType.WORD:
            yield from self._split_words(text, metadata_extractors, text_language)
        elif unit_type == TokenType.SENTENCE:
            yield from self._split_sentences(text, metadata_extractors, text_language)
        elif unit_type == TokenType.PARAGRAPH:
            yield from self._split_paragraphs(text, metadata_extractors, text_language)
        elif unit_type == TokenType.DOCUMENT:
            yield from self._split_document(text, metadata_extractors, text_language)
        else:
            raise ValueError(f"Unsupported unit type: {unit_type}")

    def _split_words(
        self, 
        text: str, 
        metadata_extractors: Dict[str, callable],
        language: Language
    ) -> Iterator[TextUnit]:
        """Split text into words based on language."""
        if not text.strip():
            return

        if language == Language.CHINESE:
            word_iterator = self._split_words_chinese(text)
        elif language == Language.JAPANESE:
            word_iterator = self._split_words_by_char(text)
        elif language == Language.MIXED:
            word_iterator = self._split_words_mixed(text)
        else:  # Language.ENGLISH
            pattern = self._word_pattern or re.compile(r'\b[a-zA-Z]+\b')
            word_iterator = ((match.group(), match.start(), match.end()) 
                           for match in pattern.finditer(text))

        for word, start, end in word_iterator:
            if word.strip():
                metadata = self._extract_metadata(word, metadata_extractors)
                yield TextUnit(
                    content=word,
                    unit_type=TokenType.WORD,
                    start_pos=start,
                    end_pos=end,
                    metadata=metadata,
                    language=language
                )

    def _split_document(
        self, 
        text: str, 
        metadata_extractors: Dict[str, callable],
        language: Language
    ) -> Iterator[TextUnit]:
        """Handle document as a single unit."""
        if not text.strip():
            return
            
        metadata = self._extract_metadata(text, metadata_extractors)
        yield TextUnit(
            content=text,
            unit_type=TokenType.DOCUMENT,
            start_pos=0,
            end_pos=len(text),
            metadata=metadata,
            language=language
        )

    def _extract_metadata(
        self, 
        text: str, 
        extractors: Dict[str, callable]
    ) -> Dict[str, Any]:
        """Extract metadata using provided extractor functions."""
        metadata = {}
        for key, extractor in extractors.items():
            try:
                metadata[key] = extractor(text)
            except Exception as e:
                metadata[key] = None
        return metadata

    def set_patterns(
        self,
        sentence_pattern: str = None,
        word_pattern: str = None,
        paragraph_pattern: str = None
    ) -> None:
        """Update the patterns used for text splitting."""
        if sentence_pattern:
            self._sent_pattern = re.compile(sentence_pattern)
            
        if word_pattern:
            self._word_pattern = re.compile(word_pattern)
            
        if paragraph_pattern:
            self._para_pattern = re.compile(paragraph_pattern)