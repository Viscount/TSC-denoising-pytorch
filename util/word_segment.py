#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba.posseg as segtool
import re

ACCEPTABLE_TYPE = {'n', 't', 's', 'f', 'v', 'a', 'b', 'z', 'e', 'y', 'o'}
REJECT_TYPE = {'u'}
REPLACE_DICT = {
    "233+": "233",
    "666+": "666",
    "emm+": "emm",
    "hhh+": "hhh",
    "www+": "www"
}


def check_type(word_type, mode):
    if mode == 'ACPT':
        if word_type[0] in ACCEPTABLE_TYPE:
            return True
        else:
            return False
    elif mode == 'REJT':
        if word_type[0] in REJECT_TYPE:
            return False
        else:
            return True
    else:
        return True


def check_replace(word):
    for item in REPLACE_DICT.keys():
        pattern = re.compile(item)
        if re.match(pattern, word) is not None:
            new_word = REPLACE_DICT[item]
            return new_word
    return word


def word_segment(content):
    words = []
    results = segtool.cut(content)
    for result in results:
        result.word = check_replace(result.word)
        if check_type(result.flag, "REJT"):
            words.append(result.word)
    return words
