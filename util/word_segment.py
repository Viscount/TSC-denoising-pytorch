#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba.posseg as segtool
import re
from xpinyin import Pinyin

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


def word_segment(content, mode='ch'):
    words = []
    results = segtool.cut(content)
    for result in results:
        result.word = check_replace(result.word)
        if check_type(result.flag, "REJT"):
            words.append(result.word)
    if mode == 'ch':
        return words
    elif mode == 'py':
        p = Pinyin()
        words_ = [p.get_pinyin(word, '') for word in words]
        return words_
    else:
        return words

if __name__ == "__main__":
    p = Pinyin()
    word = p.get_pinyin("你好", '')
    print(word)
    print(type(word))
