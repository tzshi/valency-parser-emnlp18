#!/usr/bin/env python
# encoding: utf-8

ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)

EPSILON=1e-10

UNKPART = "*UNK*"

SETTINGS = {
    "default": {
        "cat_layers": [],
        "label_mapping": {},
    },
    "udcore": {
        "cat_layers": [
            {"nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp"},
        ],
        "label_mapping": {
        },
    },
    "udfunc": {
        "cat_layers": [
            {"det", "aux", "cop", "case", "mark", "clf"},
        ],
        "label_mapping": {
        },
    },
    "udobl": {
        "cat_layers": [
            {"nmod", "obl"},
        ],
        "label_mapping": {
        },
    },
    "udcoreobl": {
        "cat_layers": [
            {"nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp"},
            {"nmod", "obl"},
        ],
        "label_mapping": {
        },
    },
    "udcorefunc": {
        "cat_layers": [
            {"nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp"},
            {"det", "aux", "cop", "case", "mark", "clf"},
        ],
        "label_mapping": {
        },
    },
    "udcorefuncobl": {
        "cat_layers": [
            {"nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp"},
            {"det", "aux", "cop", "case", "mark", "clf"},
            {"nmod", "obl"},
        ],
        "label_mapping": {
        },
    },
    "ptb": {
        "cat_layers": [
            {"agent", "acomp", "ccomp", "xcomp", "dobj", "iobj", "pobj", "nsubj", "csubj", "csubjpass", "nsubjpass"},
            {"aux", "auxpass", "cop", "mark", "det", "predet", "prt", "possessive", "expl"},
        ],
        "label_mapping": {
        },
    },
    "tag": {
        "cat_layers": [
            {"0", "1", "2"},
            {"CO"},
        ],
        "label_mapping": {
        },
    },
}

VALENCY = {}


def set_part_mode(mode):
    global VALENCY
    global SETTINGS
    VALENCY["cat_layers"] = SETTINGS[mode]["cat_layers"]
    VALENCY["label_mapping"] = SETTINGS[mode]["label_mapping"]
    if len(VALENCY["cat_layers"]) > 0:
        VALENCY["core"] = set.union(*VALENCY["cat_layers"])
    else:
        VALENCY["core"] = set()
