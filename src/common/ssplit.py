#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sentence boundary generator for BRAT / BIO conversion.
"""

from re import compile as re_compile
from re import DOTALL, VERBOSE

# Reasonably well-behaved sentence end regular expression
SENTENCE_END_REGEX = re_compile(r'''
        # Require a leading non-whitespace character for the sentence
        \S
        # Then, anything goes, but don't be greedy
        .*?
        # Anchor the sentence at...
        (:?
            # One (or multiple) terminal character(s)
            #   followed by one (or multiple) whitespace
            (:?(\.|!|\?|。|！|？)+(?=\s+))
        | # Or...
            # Newlines, to respect file formatting
            (:?(?=\n+))
        | # Or...
            # End-of-file, excluding whitespaces before it
            (:?(?=\s*$))
        )
    ''', DOTALL | VERBOSE)

SENTENCE_END_NEWLINE_REGEX = re_compile(r'''
        # Require a leading non-whitespace character for the sentence
        \S
        # Then, anything goes, but don't be greedy
        .*?
        # Anchor the sentence at...
        (:?
            # One (or multiple) newlines
            (:?(?=\n+))
        | # Or...
            # End-of-file, excluding whitespaces before it
            (:?(?=\s*$))
        )
    ''', DOTALL | VERBOSE)


def _refine_split(offsets, original_text):
    new_text = '\n'.join((original_text[o[0]:o[1]].replace('\n', ' ')
                          for o in offsets))

    try:
        from sspostproc import refine_split
        output = refine_split(new_text)
    except ImportError:
        output = new_text

    old_offsets = offsets[::-1]
    if len(old_offsets) == 0:
        old_offsets.append((0, len(original_text), ))
    new_offsets = []
    for refined_sentence in output.split('\n'):
        if not old_offsets:
            break
        new_offset = old_offsets.pop()
        while new_offset[1] - new_offset[0] < len(refined_sentence) - 1 and old_offsets:
            _, next_end = old_offsets.pop()
            new_offset = (new_offset[0], next_end)
        new_offsets.append(new_offset)

    if len(new_offsets) != 0 and new_offsets[-1][1] != len(original_text) - 1:
        start = new_offsets[-1][1] + 1
        while start < len(original_text) and original_text[start].isspace():
            start += 1
        if start < len(original_text) - 1:
            new_offsets.append((start, len(original_text) - 1))

    last_newline = -1
    while True:
        try:
            orig_newline = original_text.index('\n', last_newline + 1)
        except ValueError:
            break

        for o_start, o_end in new_offsets:
            if o_start <= orig_newline < o_end:
                new_offsets.remove((o_start, o_end))
                new_offsets.extend(((o_start, orig_newline, ),
                                    (orig_newline + 1, o_end), ))
                break
            elif o_end == orig_newline:
                break
        else:
            new_offsets.append((orig_newline, orig_newline, ))

        last_newline = orig_newline

    new_offsets.sort()
    return new_offsets


def _sentence_boundary_gen(text, regex):
    for match in regex.finditer(text):
        yield match.span()


def regex_sentence_boundary_gen(text):
    for o in _refine_split([_o for _o in _sentence_boundary_gen(text, SENTENCE_END_REGEX)], text):
        yield o


def newline_sentence_boundary_gen(text):
    for o in _sentence_boundary_gen(text, SENTENCE_END_NEWLINE_REGEX):
        yield o
