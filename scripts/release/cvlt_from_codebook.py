"""Rebuild the CVLT-II Standard source for the paper build from the public UCLA NNN codebook.

The word lists are Pearson copyright and are not part of this repository. The codebook
(https://www.nnn.ucla.edu/downloads/codebook.pdf, pp. 384-386) prints the 48 yes/no recognition
items in order. paper_materials/cvlt_structure.json holds everything else the build needs without
any list word: for each List A and List B position, the recognition item that holds its word, plus
categories, item types, and the administration text. This module joins the two on your machine.
"""
import json
from pathlib import Path
import re

STRUCTURE = Path(__file__).resolve().parent / 'paper_materials' / 'cvlt_structure.json'
CODEBOOK = 'ucla_cnp_ds000030/nnn_codebook.pdf'
ITEM = re.compile(r'List A Long-Delay Yes/No Recognition:[ \t]*([A-Za-z]+)[ \t]*$', re.M)
SUMMARY = {'total'}


def recognition_words(pdf_path):
    """The 48 yes/no recognition words, in codebook order."""
    import fitz
    with fitz.open(pdf_path) as document:
        text = '\n'.join(page.get_text() for page in document).replace('\xa0', ' ')  # the codebook uses no-break spaces
    words = [w.lower() for w in ITEM.findall(text) if w.lower() not in SUMMARY]
    if len(words) != 48:
        raise ValueError(f'Expected 48 CVLT-II recognition items in {pdf_path}, found {len(words)}')
    return words


def build_source(raw_root, structure_path=STRUCTURE):
    structure = json.loads(Path(structure_path).read_text())
    words = recognition_words(Path(raw_root) / CODEBOOK)
    source = {k: v for k, v in structure.items() if k not in ('note', 'lists', 'recognition')}
    source['source'] = {**structure['source'], 'material_path': str(Path(raw_root) / CODEBOOK)}
    source['lists'] = {name: [{'word': words[item['recognition_position'] - 1], 'category': item['category']}
                              for item in items] for name, items in structure['lists'].items()}
    source['recognition'] = [{'position': item['position'], 'word': words[item['position'] - 1],
                              **{k: v for k, v in item.items() if k != 'position'}}
                             for item in structure['recognition']]
    ordered = {}
    for key in structure['key_order']:
        ordered[key] = source[key]
    return ordered
