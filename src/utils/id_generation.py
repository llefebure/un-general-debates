"""Unique ID generation"""
import hashlib

def generate_document_id(text):
    """Generates a unique and deterministic id based on the input text."""
    m = hashlib.md5()
    m.update(text.encode('utf-8'))
    return str(int(m.hexdigest(), 16))[:8]
