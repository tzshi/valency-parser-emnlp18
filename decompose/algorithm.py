#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from .const import VALENCY

NEGINF = -np.inf


def get_layer_labels(graph, pos):
    ret = []
    for CATS in VALENCY["cat_layers"]:
        part = []
        for n, (h, l) in enumerate(zip(graph.heads, graph.rels)):
            if h == pos:
                if l in CATS:
                    part.append(VALENCY["label_mapping"].get(l, l))
            if n == pos:
                part.append("*")
        ret.append(" ".join(part))
    return ret


def get_part_label(graph, pos):
    part = []
    for n, (h, l) in enumerate(zip(graph.heads, graph.rels)):
        if h == pos:
            if l in VALENCY["core"]:
                part.append(VALENCY["label_mapping"].get(l, l))
        if n == pos:
            part.append("*")

    return " ".join(part)


def extract_parts(graph, pos, ret):
    part = []
    for n, (h, l) in enumerate(zip(graph.heads, graph.rels)):
        if h == pos:
            if l in VALENCY["core"]:
                part.append(l)
            extract_parts(graph, n, ret)
        if n == pos and n > 0:
            part.append("*")

    if len(part) > 0:
        ret.append(" ".join(part))
