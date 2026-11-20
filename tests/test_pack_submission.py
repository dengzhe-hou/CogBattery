import pytest

from scripts.release.pack_submission import decode_words, encode_words

# Made-up words that stand in for a study list. The real CVLT-II words never appear in the repository.
WORDS = sorted(['zorvak', 'plimet', 'quorbel'])


def test_list_words_are_encoded_and_restored_exactly():
    record = {'response': '["Zorvak", "PLIMET", "quorbels", "zorvaks"]',
              'raw_content': 'I remember: zorvak, plimets, QuOrBeL, zorvakfish.', 'usage': {'eval_count': 7},
              'protocol_errors': ['plimet was repeated']}
    packed = encode_words(record, WORDS)
    # A longer word that only starts with a list word is a different word and stays as it is.
    assert 'zorvakfish' in packed['raw_content']
    text = repr(packed).lower().replace('zorvakfish', '')
    assert not any(w in text for w in WORDS)
    assert packed['usage'] == {'eval_count': 7}
    assert decode_words(packed, WORDS) == record


def test_no_list_word_is_left_in_a_packed_reply():
    packed = encode_words('Quorbel, plimet and zorvaks.', WORDS)
    assert not any(w in packed.lower() for w in WORDS)


def test_reply_with_the_token_brackets_is_refused():
    with pytest.raises(SystemExit):
        encode_words('⟦0⟧', WORDS)


def test_token_that_does_not_fit_the_materials_is_refused():
    with pytest.raises(SystemExit):
        decode_words('⟦0:UU⟧', WORDS)
