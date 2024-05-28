from num2words import num2words
import nltk
import typing
import re
import numpy as np

PHONEMES_TO_VISEMES = {
    "B": ["p"],
    "D": ["t"],
    "G": ["k"],
    "P": ["p"],
    "T": ["t"],
    "K": ["k"],
    "JH": ["ch"],
    "CH": ["ch"],
    "S": ["t"],
    "SH": ["ch"],
    "Z": ["t"],
    "ZH": ["ch"],
    "F": ["f"],
    "TH": ["t"],
    "V": ["f"],
    "DH": ["t"],
    "M": ["p"],
    "N": ["k"],
    "NG": ["k"],
    "L": ["k"],
    "R": ["w"],
    "W": ["w"],
    "Y": ["k"],
    "HH": ["k"],
    "IY": ["iy"],
    "IH": ["iy"],
    "EH": ["eh"],
    "EY": ["eh", "iy"],  # Note multiple visemes given to a single phoneme in some cases
    "AE": ["eh"],
    "AA": ["aa"],
    "AW": ["aa", "uh"],
    "AY": ["aa", "iy"],
    "AH": ["ah"],
    "AO": ["ao"],
    "OY": ["ao", "iy"],
    "OW": ["ao", "uh"],
    "UH": ["uh"],
    "UW": ["uh"],
    "ER": ["er"],
}
BAD_WORD_KEY: str = "BADWORD"


def word_to_phonemes(word: str) -> typing.List[str]:
    """
    Converts a word to a list of phonemes
    """
    try:
        arpabet = nltk.corpus.cmudict.dict()
    except:
        nltk.download("cmudict")
        arpabet = nltk.corpus.cmudict.dict()
    word = word.lower()
    if word.isdigit():
        if word == "000":
            word = "thousand"
        else:
            word = num2words(word)

    if word not in arpabet.keys():
        return [BAD_WORD_KEY]
    return arpabet[word][0]


def phonemes_to_visemes(phonemes: typing.List[str]) -> typing.List[str]:
    """
    Converts a list of phonemes and then to a list of visemes
    """
    visemes = []
    for phoneme_subset in [
        PHONEMES_TO_VISEMES[re.sub("[0-9]", "", x)] for x in phonemes
    ]:
        visemes.extend(phoneme_subset)
    return visemes
