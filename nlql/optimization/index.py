from typing import Dict, Set, Any
import re
from collections import defaultdict
from ..utils.text_unit import TextUnit, Language

class TextIndex:
    """
    Text indexing system to optimize query performance.
    Provides inverted index for word-based searches and metadata indices.
    """
    
    def __init__(self):
        """Initialize text indices."""
        # Word-based inverted index
        self.word_index: Dict[str, Set[int]] = defaultdict(set)
        
        # Metadata indices
        self.metadata_indices: Dict[str, Dict[Any, Set[int]]] = defaultdict(lambda: defaultdict(set))
        
        # Position mapping for text units
        self.position_map: Dict[int, TextUnit] = {}
        
        # Counter for assigning positions
        self._position_counter = 0

    def add_unit(self, unit: TextUnit):
        """
        Add a text unit to the index.
        
        Args:
            unit (TextUnit): Text unit to index
        """
        position = self._position_counter
        self.position_map[position] = unit
        
        # Index words
        words = self._extract_words(unit.content, unit.language)
        for word in words:
            self.word_index[word.lower()].add(position)
        
        # Index metadata
        for key, value in unit.metadata.items():
            if value is not None:  # Only index non-None values
                self.metadata_indices[key][value].add(position)
        
        self._position_counter += 1

    def _extract_words(self, text: str, language: Language) -> Set[str]:
        """Extract words from text based on language."""
        if language == Language.CHINESE:
            try:
                import jieba
                return set(jieba.cut(text))
            except ImportError:
                # Fallback to character splitting for Chinese
                return set(text)
        else:
            # Simple word extraction for other languages
            return set(re.findall(r'\w+', text.lower()))

    def search_word(self, word: str) -> Set[TextUnit]:
        """
        Search for text units containing a word.
        
        Args:
            word (str): Word to search for
            
        Returns:
            Set[TextUnit]: Text units containing the word
        """
        positions = self.word_index.get(word.lower(), set())
        return {self.position_map[pos] for pos in positions}

    def search_by_metadata(self, key: str, value: Any) -> Set[TextUnit]:
        """
        Search for text units by metadata value.
        
        Args:
            key (str): Metadata key
            value (Any): Value to search for
            
        Returns:
            Set[TextUnit]: Text units matching the metadata value
        """
        positions = self.metadata_indices.get(key, {}).get(value, set())
        return {self.position_map[pos] for pos in positions}

    def search_range(self, key: str, start: Any, end: Any) -> Set[TextUnit]:
        """
        Search for text units with metadata values in a range.
        
        Args:
            key (str): Metadata key
            start (Any): Range start (inclusive)
            end (Any): Range end (inclusive)
            
        Returns:
            Set[TextUnit]: Text units in the range
        """
        results = set()
        if key in self.metadata_indices:
            for value, positions in self.metadata_indices[key].items():
                if start <= value <= end:
                    results.update(self.position_map[pos] for pos in positions)
        return results

    def clear(self):
        """Clear all indices."""
        self.word_index.clear()
        self.metadata_indices.clear()
        self.position_map.clear()
        self._position_counter = 0