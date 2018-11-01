#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from collections import Counter
import re
import ftfy
import random
import numpy as np
from .const import UNKPART
from .const import VALENCY


def normalize(word):
    return re.sub(r"\d", "0", word).lower()


def strong_normalize(word):
    w = ftfy.fix_text(word.lower())
    w = re.sub(r".+@.+", "*EMAIL*", w)
    w = re.sub(r"@\w+", "*AT*", w)
    w = re.sub(r"(https?://|www\.).*", "*url*", w)
    w = re.sub(r"([^\d])\1{2,}", r"\1\1", w)
    w = re.sub(r"([^\d][^\d])\1{2,}", r"\1\1", w)
    w = re.sub(r"``", '"', w)
    w = re.sub(r"''", '"', w)
    w = re.sub(r"\d", "0", w)
    return w


def buildVocab(graphs, cutoff=1, partcutoff=1, casecutoff=50):
    CAT_LAYERS = VALENCY["cat_layers"]
    CORE = VALENCY["core"]
    LABEL_MAPPING = VALENCY["label_mapping"]

    wordsCount = Counter()
    charsCount = Counter()
    uposCount = Counter()
    xposCount = Counter()
    casesCount = Counter()
    relCount = Counter()
    featCount = Counter()
    partCount = Counter()
    layerCount = [Counter() for _ in CAT_LAYERS]

    for graph in graphs:
        wordsCount.update([node.norm for node in graph.nodes[1:]])
        for node in graph.nodes[1:]:
            charsCount.update(list(node.word))
            featCount.update(node.feats_set)
        uposCount.update([node.upos for node in graph.nodes[1:]])
        xposCount.update([node.xupos for node in graph.nodes[1:]])
        casesCount.update([node.cases for node in graph.nodes[1:] if node.cases is not None])
        relCount.update([rel for rel in graph.rels[1:]])
        partCount.update([node.part for node in graph.nodes[1:]])
        for i in range(len(CAT_LAYERS)):
            layerCount[i].update([node.layers[i] for node in graph.nodes[1:]])

    print("Number of tokens in training corpora: {}".format(sum(wordsCount.values())))
    print("Vocab containing {} types before cutting off".format(len(wordsCount)))
    wordsCount = Counter({w: i for w, i in wordsCount.items() if i >= cutoff})
    partCount = Counter({p: i for p, i in partCount.items() if i >= partcutoff})
    casesCount = Counter({p: i for p, i in casesCount.items() if i >= casecutoff})
    print("A total of {} layers".format(len(CAT_LAYERS)))
    for i in range(len(CAT_LAYERS)):
        print("Layer {} containing {} tags before cutting off".format(i, len(layerCount[i])))
        layerCount[i] = Counter({l: j for l, j in layerCount[i].items() if j >= partcutoff})
    print("Vocab containing {} types, covering {} words".format(len(wordsCount), sum(wordsCount.values())))
    print("Charset containing {} chars".format(len(charsCount)))
    print("UPOS containing {} tags".format(len(uposCount)), uposCount)
    print("XPOS containing {} tags".format(len(xposCount)))
    print("CASES containing {} tags".format(len(casesCount)), casesCount)
    print("Rels containing {} tags".format(len(relCount)), relCount)
    print("Feats containing {} tags".format(len(featCount)))
    print("A total of {} layers".format(len(CAT_LAYERS)))
    for i in range(len(CAT_LAYERS)):
        print("Layer {} containing {} tags, covering {} words".format(i, len(layerCount[i]), sum(layerCount[i].values())), layerCount[i])

    rel_cats = set()
    for label in CORE:
        if label in relCount:
            if label in LABEL_MAPPING:
                rel_cats.add(LABEL_MAPPING[label])
            else:
                rel_cats.add(label)
    for label in relCount:
        if label not in CORE:
            rel_cats.add("_noncore")
            break

    ret = {
        "vocab": list(wordsCount.keys()),
        "wordfreq": wordsCount,
        "charset": list(charsCount.keys()),
        "charfreq": charsCount,
        "upos": list(uposCount.keys()),
        "xpos": list(xposCount.keys()),
        "cases": [UNKPART] + list(casesCount.keys()),
        "rels": list(relCount.keys()),
        "relcats": list(rel_cats),
        "feats": list(featCount.keys()),
        "part": [UNKPART] + list([x[0] for x in partCount.most_common()]),
        "layers": [[UNKPART] + list([x[0] for x in layer.most_common()]) for layer in layerCount],
        "label_mapping": LABEL_MAPPING,
        "core": list(sorted(CORE)),
        "cat_layers": [list(sorted(layer)) for layer in CAT_LAYERS]
    }

    return ret


def build_part_mappings(parts, relcats):
    ret = [[-1, -1, -1, -1] for part in parts]
    ret2 = [[-1, -1] for part in parts]
    next_i = len(parts)
    for i, part in enumerate(parts):
        if part == UNKPART:
            ret[i] = [-i -1, -i - 1, 0, 0]
            ret2[i] = [-i - 1, -i - 1]
            continue
        p = part.split()
        star = p.index("*")
        left_num = star
        right_num = len(p) - star - 1
        ret[i][0] = next_i
        for j in range(left_num):
            next_i += 1
            ret.append([-1, -1, -1, -1])
            ret[next_i - 1][0] = next_i
            if j < left_num - 1:
                ret[next_i - 1][2] = relcats[p[star - j - 2]]
        left_end = next_i - 1 if left_num > 0 else i

        ret[i][1] = next_i
        for j in range(right_num):
            next_i += 1
            ret.append([-1, -1, -1, -1])
            ret[next_i - 1][1] = next_i
            if j < right_num - 1:
                ret[next_i - 1][3] = relcats[p[star + j + 2]]
        right_end = next_i -1 if right_num > 0 else i

        ret[left_end][0] = -i - 1
        ret[right_end][1] = -i - 1
        ret2[i][0] = left_end
        ret2[i][1] = right_end
        if left_num > 0:
            ret[i][2] = relcats[p[star - 1]]
        if right_num > 0:
            ret[i][3] = relcats[p[star + 1]]

    return np.array(ret, dtype=np.int32), np.array(ret2, dtype=np.int32)


def shuffled_stream(data):
    len_data = len(data)
    while True:
        for d in random.sample(data, len_data):
            yield d


def shuffled_balanced_stream(data):
    for ds in zip(*[shuffled_stream(s) for s in data]):
        ds = list(ds)
        random.shuffle(ds)
        for d in ds:
            yield d
