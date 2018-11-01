#!/usr/bin/env python
# encoding: utf-8

from dynet import *
from collections import Counter
import random

from . import pyximportcpp; pyximportcpp.install()
from .calgorithm import parse_proj
from .astarhsel import astar_hsel, astar_hsel_new, chart_hsel, chart_hsel_supertag, best_label_per_cat, astar_hsel_supertag_new, astar_hsel_supertag_multi
from .astarhsel import astar_hsel_supertag_multi_lexlimit

from .utils import build_part_mappings

from .layers import MultiLayerPerceptron, Dense, Bilinear, identity, Biaffine, BiaffineBatch
from .const import EPSILON

class Labeler:

    def __init__(self, parser, id="Labeler", **kwargs):
        self._parser = parser
        self.id = id

        self._bilstm_dims = kwargs.get("bilstm_dims", 128)
        self._label_mlp_activation = self._parser._activations[kwargs.get('label_mlp_activation', 'relu')]
        self._label_mlp_dims = kwargs.get("label_mlp_dims", 128)
        self._label_mlp_layers = kwargs.get("label_mlp_layers", 1)
        self._label_mlp_dropout = kwargs.get("label_mlp_dropout", 0.0)

    def init_params(self):
        self._label_head_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._label_mlp_dims] * self._label_mlp_layers, self._label_mlp_activation, self._parser._model)
        self._label_mod_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._label_mlp_dims] * self._label_mlp_layers, self._label_mlp_activation, self._parser._model)
        self._label_scorer = BiaffineBatch(self._label_mlp_dims, len(self._parser._rels), self._parser._model)

    def init_cg(self, train=False):
        if train:
            self._label_head_mlp.set_dropout(self._label_mlp_dropout)
            self._label_mod_mlp.set_dropout(self._label_mlp_dropout)
        else:
            self._label_head_mlp.set_dropout(0.)
            self._label_mod_mlp.set_dropout(0.)

    def _label_arcs_eval(self, carriers):
        vecs = concatenate([c. vec for c in carriers], 1)
        expr = self._label_scorer(self._label_head_mlp(vecs), self._label_mod_mlp(vecs))

        expr = reshape(expr, (len(carriers) * len(carriers), len(self._parser._rels)))
        expr = transpose(log_softmax(transpose(expr)))
        expr = reshape(expr, (len(carriers), len(carriers), len(self._parser._rels)))

        return expr

    def sent_loss(self, graph, carriers):
        correct = 0
        loss = []
        exprs = self._label_arcs_eval(carriers)

        for mod, head in enumerate(graph.heads):
            if mod > 0 and head >= 0:
                if not graph.rels[mod] in self._parser._rels:
                    continue

                answer = self._parser._rels[graph.rels[mod]]

                loss.append(-exprs[int(head)][int(mod)][int(answer)])

        return correct, loss

    def predict(self, graph, carriers):
        exprs = self._label_arcs_eval(carriers)
        scores = exprs.npvalue()
        for mod, head in enumerate(graph.heads):
            if mod > 0 and head >= 0:
                graph.rels[mod] = self._parser._irels[np.argmax(scores[head, mod])]

        return self


class UPOSTagger:

    def __init__(self, parser, id="UPOSTagger", **kwargs):
        self._parser = parser
        self.id = id

        self._bilstm_dims = kwargs.get("bilstm_dims", 128)
        self._utagger_mlp_activation = self._parser._activations[kwargs.get('utagger_mlp_activation', 'relu')]
        self._utagger_mlp_dims = kwargs.get("utagger_mlp_dims", 128)
        self._utagger_mlp_layers = kwargs.get("utagger_mlp_layers", 2)
        self._utagger_mlp_dropout = kwargs.get("utagger_mlp_dropout", 0.0)

    def init_params(self):
        self._utagger_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._utagger_mlp_dims] * self._utagger_mlp_layers, self._utagger_mlp_activation, self._parser._model)
        self._utagger_final = Dense(self._utagger_mlp_dims, len(self._parser._upos), identity, self._parser._model)

    def init_cg(self, train=False):
        if train:
            self._utagger_mlp.set_dropout(self._utagger_mlp_dropout)
        else:
            self._utagger_mlp.set_dropout(0.)

    def sent_loss(self, graph, carriers):
        ret = []
        correct = 0

        for c in carriers[1:]:
            potentials = self._utagger_final(self._utagger_mlp(c.vec))
            answer = self._parser._upos[c.node.upos]

            ret.append(pickneglogsoftmax(potentials, answer))
        return correct, ret

    def predict(self, graph, carriers):
        for c in carriers[1:]:
            potentials = self._utagger_final(self._utagger_mlp(c.vec))
            pred = np.argmax(potentials.npvalue())
            c.node.upos = self._parser._iupos[pred]

        return self

class XPOSTagger:
    def __init__(self, parser, id="XPOSTagger", **kwargs):
        self._parser = parser
        self.id = id

        self._bilstm_dims = kwargs.get("bilstm_dims", 128)
        self._xtagger_mlp_activation = self._parser._activations[kwargs.get('xtagger_mlp_activation', 'relu')]
        self._xtagger_mlp_dims = kwargs.get("xtagger_mlp_dims", 128)
        self._xtagger_mlp_layers = kwargs.get("xtagger_mlp_layers", 2)
        self._xtagger_mlp_dropout = kwargs.get("xtagger_mlp_dropout", 0.0)

    def init_params(self):
        self._xtagger_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._xtagger_mlp_dims] * self._xtagger_mlp_layers, self._xtagger_mlp_activation, self._parser._model)
        self._xtagger_final = Dense(self._xtagger_mlp_dims, len(self._parser._xpos), identity, self._parser._model)

    def init_cg(self, train=False):
        if train:
            self._xtagger_mlp.set_dropout(self._xtagger_mlp_dropout)
        else:
            self._xtagger_mlp.set_dropout(0.)

    def sent_loss(self, graph, carriers):
        ret = []
        correct = 0

        for c in carriers[1:]:
            potentials = self._xtagger_final(self._xtagger_mlp(c.vec))
            answer = self._parser._xpos[c.node.xupos]

            ret.append(pickneglogsoftmax(potentials, answer))
        return correct, ret

    def predict(self, graph, carriers):
        for c in carriers[1:]:
            potentials = self._xtagger_final(self._xtagger_mlp(c.vec))
            pred = np.argmax(potentials.npvalue())
            c.node.xpos = self._parser._ixpos[pred].split("|")[1]

        return self


class PartTagger:
    def __init__(self, parser, id="PartTagger", **kwargs):
        self._parser = parser
        self.id = id

        self._bilstm_dims = kwargs.get("bilstm_dims", 128)
        self._parttagger_mlp_activation = self._parser._activations[kwargs.get('parttagger_mlp_activation', 'relu')]
        self._parttagger_mlp_dims = kwargs.get("parttagger_mlp_dims", 128)
        self._parttagger_mlp_layers = kwargs.get("parttagger_mlp_layers", 2)
        self._parttagger_mlp_dropout = kwargs.get("parttagger_mlp_dropout", 0.0)

    def init_params(self):
        self._parttagger_mlps = [MultiLayerPerceptron([self._bilstm_dims] + [self._parttagger_mlp_dims] * self._parttagger_mlp_layers, self._parttagger_mlp_activation, self._parser._model) for _ in self._parser._layers]
        self._parttagger_finals = [Dense(self._parttagger_mlp_dims, len(layer), identity, self._parser._model) for layer in self._parser._layers]

    def init_cg(self, train=False):
        for parttagger_mlp in self._parttagger_mlps:
            if train:
                parttagger_mlp.set_dropout(self._parttagger_mlp_dropout)
            else:
                parttagger_mlp.set_dropout(0.)

    def prob_matrix(self, carriers):
        ret = []
        for mlp, final, layer in zip(self._parttagger_mlps, self._parttagger_finals, self._parser._layers):
            _ret = np.zeros((len(carriers), len(layer)))
            for i, c in enumerate(carriers):
                potentials = final(mlp(c.vec))
                _ret[i, :] = log_softmax(potentials).npvalue()
            ret.append(_ret)
        return np.concatenate(ret, axis=1)

    def sent_loss(self, graph, carriers):
        ret = []
        correct = 0

        for i in range(len(self._parser._layers)):
            mlp = self._parttagger_mlps[i]
            final = self._parttagger_finals[i]
            layer = self._parser._layers[i]
            for c in carriers:
                potentials = final(mlp(c.vec))
                answer = layer.get(c.node.layers[i], 0)

                ret.append(pickneglogsoftmax(potentials, answer))
        return correct, ret

    def predict(self, graph, carriers):
        for c in carriers[1:]:
            c.node.layers = []
            for mlp, final, ilayer in zip(self._parttagger_mlps, self._parttagger_finals, self._parser._ilayers):
                potentials = final(mlp(c.vec))
                pred = np.argmax(potentials.npvalue())
                c.node.layers.append(ilayer[pred])

        return self


class HSelParser:
    def __init__(self, parser, id="HSelParser", **kwargs):
        self._parser = parser
        self.id = id

        self._bilstm_dims = kwargs.get("bilstm_dims", 128)
        self._hsel_mlp_activation = self._parser._activations[kwargs.get('hsel_mlp_activation', 'relu')]
        self._hsel_mlp_dims = kwargs.get("hsel_mlp_dims", 128)
        self._hsel_mlp_layers = kwargs.get("hsel_mlp_layers", 2)
        self._hsel_mlp_dropout = kwargs.get("hsel_mlp_dropout", 0.0)
        self._hsel_postprocess = kwargs.get("hsel_postprocess", "none")

    def init_params(self):
        self._hsel_head_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._hsel_mlp_dims] * self._hsel_mlp_layers, self._hsel_mlp_activation, self._parser._model)
        self._hsel_mod_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._hsel_mlp_dims] * self._hsel_mlp_layers, self._hsel_mlp_activation, self._parser._model)
        self._hsel_bilinear = Bilinear(self._hsel_mlp_dims, self._parser._model)
        self._hsel_head_bias = Dense(self._hsel_mlp_dims, 1, identity, self._parser._model)

    def init_cg(self, train=False):
        if train:
            self._hsel_head_mlp.set_dropout(self._hsel_mlp_dropout)
            self._hsel_mod_mlp.set_dropout(self._hsel_mlp_dropout)
        else:
            self._hsel_head_mlp.set_dropout(0.)
            self._hsel_mod_mlp.set_dropout(0.)

    def _hsel_arcs_eval(self, carriers):
        vecs = concatenate([c.vec for c in carriers], 1)
        head_vecs = self._hsel_head_mlp(vecs)
        mod_vecs = self._hsel_mod_mlp(vecs)

        exprs = colwise_add(self._hsel_bilinear(head_vecs, mod_vecs), reshape(self._hsel_head_bias(head_vecs), (len(carriers),)))
        exprs = log_softmax(exprs)

        return exprs

    def sent_loss(self, graph, carriers):
        gold_heads = graph.heads

        exprs = self._hsel_arcs_eval(carriers)

        loss = [-(exprs[int(h)][int(m)]) for m, h in enumerate(gold_heads) if m > 0]

        return 0, loss

    def predict(self, graph, carriers):
        exprs = self._hsel_arcs_eval(carriers)
        scores = exprs.npvalue()
        total, graph.heads = astar_hsel_new(scores)

        return self


class HSelPartParser:
    def __init__(self, parser, hselparser, parttagger, labeler, id="HSelPartParser", **kwargs):
        self._parser = parser
        self.id = id

        self._hselparser = hselparser
        self._parttagger = parttagger
        self._labeler = labeler

    def init_params(self):
        self._layer_index = [0]
        for l in self._parser._layers:
            self._layer_index.append(self._layer_index[-1] + len(l))
        self._layer_index = np.array(self._layer_index, dtype=np.int32)
        return

    def init_cg(self, train=False):
        return

    def sent_loss(self, graph, carriers):
        return 0., []

    def predict(self, graph, carriers):
        exprs = self._hselparser._hsel_arcs_eval(carriers)
        scores = exprs.npvalue()

        supertag = self._parttagger.prob_matrix(carriers)

        exprs = self._labeler._label_arcs_eval(carriers)
        label_scores = exprs.npvalue()
        label, best_labels = best_label_per_cat(label_scores, self._parser._rel2cat, len(self._parser._irelcats) + 1, self._parser._relcats["_noncore"])

        total, graph.heads, rels = astar_hsel_supertag_multi_lexlimit(scores, supertag, label, self._parser._layermappings, best_labels, self._layer_index, self._parser._relcats["_noncore"], 500000, 500)
        if total < 0:
            self.failed += 1
            self._hselparser.predict(graph, carriers)
            self._labeler.predict(graph, carriers)
        else:
            for i in range(1, len(carriers)):
                graph.rels[i] = self._parser._irels[rels[i]]

        return self
