"""
Document Filters
"""
from typing import Dict, Any, List, Callable

class DocumentFilter:
    def __init__(self):
        self._filters = []
    def add(self, filter_fn):
        self._filters.append(filter_fn)
        return self
    def __call__(self, doc):
        if not self._filters: return True
        return all(f(doc) for f in self._filters)

class MinLengthFilter:
    def __init__(self, min_chars, text_column="text"):
        self.min_chars = min_chars
        self.text_column = text_column
    def __call__(self, doc):
        text = doc.get(self.text_column, "") or doc.get(list(doc.keys())[0], "")
        return len(text) >= self.min_chars

class MaxLengthFilter:
    def __init__(self, max_chars, text_column="text"):
        self.max_chars = max_chars
        self.text_column = text_column
    def __call__(self, doc):
        text = doc.get(self.text_column, "") or doc.get(list(doc.keys())[0], "")
        return len(text) <= self.max_chars