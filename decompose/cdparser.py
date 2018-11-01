#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function

import numpy as np
from collections import defaultdict
import fire, random, time, math, os, sys, json
from tqdm import tqdm
from dynet import *
from .utils import buildVocab, shuffled_balanced_stream, build_part_mappings
from .modules import UPOSTagger, XPOSTagger, PartTagger, HSelParser, Labeler, HSelPartParser

from . import pyximportcpp; pyximportcpp.install()
from .calgorithm import projectivize, is_projective
from .evaluation import POSCorrect

from .io import read_conll, write_conll
from .layers import Dense, LSTM, HighwayLSTM, leaky_relu
from .const import set_part_mode


class ComputationCarrier(object):

    def __copy__(self):
        result = object.__new__(ComputationCarrier)
        result.__dict__.update(self.__dict__)
        return result


class CDParser:

    def __init__(self, **kwargs):
        self._part_mode = kwargs.get("part_mode", "default")
        set_part_mode(self._part_mode)
        pass

    def create_parser(self, **kwargs):
        self._verbose = kwargs.get("verbose", True)
        if self._verbose:
            print("Parameters (others default):")
            for k in sorted(kwargs):
                print(k, kwargs[k])
            sys.stdout.flush()

        self._args = kwargs

        self._learning_rate = kwargs.get("learning_rate", 0.001)
        self._beta1 = kwargs.get("beta1", 0.9)
        self._beta2 = kwargs.get("beta2", 0.999)
        self._epsilon = kwargs.get("epsilon", 1e-8)
        self._edecay = kwargs.get("edecay", 0.)
        self._clip = kwargs.get("clip", 5.)
        self._sparse_updates = kwargs.get("sparse_updates", False)

        self._weight_decay = kwargs.get("weight_decay", 0.)

        self._optimizer = kwargs.get("optimizer", "adam")

        self._batch_size = kwargs.get("batch_size", 50)
        self._anneal_base = kwargs.get("anneal_base", 1.0)
        self._anneal_steps = kwargs.get("anneal_steps", 1000)

        self._word_smooth = kwargs.get("word_smooth", 0.25)
        self._char_smooth = kwargs.get("char_smooth", 0.25)

        self._wdims = kwargs.get("wdims", 128)
        self._edims = kwargs.get("edims", 0)
        self._bidirectional = kwargs.get("bidirectional", True)
        self._highway = kwargs.get("highway", False)
        if self._highway:
            self.WordLSTMBuilder = HighwayLSTM
        else:
            self.WordLSTMBuilder = LSTM
        self._bilstm_dims = kwargs.get("bilstm_dims", 128)
        self._bilstm_layers = kwargs.get("bilstm_layers", 2)
        self._bilstm_dropout = kwargs.get("bilstm_dropout", 0.0)

        self._pdims = kwargs.get("pdims", 0)
        self._fdims = kwargs.get("fdims", 0)

        self._feature_dropout = kwargs.get("feature_dropout", 0.0)

        self._block_dropout = kwargs.get("block_dropout", 0.)
        self._char_dropout = kwargs.get("char_dropout", 0.)

        self._cdims = kwargs.get("cdims", 32)
        self._char_lstm_dims = kwargs.get("char_lstm_dims", 128)
        self._char_lstm_layers = kwargs.get("char_lstm_layers", 2)
        self._char_lstm_dropout = kwargs.get("char_lstm_dropout", 0.0)

        self._char_repr_method = kwargs.get("char_repr_method", "pred")

        self._activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'lrelu': leaky_relu, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}

        self._utagger_num = kwargs.get("utagger_num", 0)
        self._utagger_weight = kwargs.get("utagger_weight", 1.0)
        self._utaggers = [UPOSTagger(self, id="UPOS-{}".format(i+1), **self._args) for i in range(self._utagger_num)]

        self._xtagger_num = kwargs.get("xtagger_num", 0)
        self._xtagger_weight = kwargs.get("xtagger_weight", 1.0)
        self._xtaggers = [XPOSTagger(self, id="XPOS-{}".format(i+1), **self._args) for i in range(self._xtagger_num)]

        self._parttagger_num = kwargs.get("parttagger_num", 0)
        self._parttagger_weight = kwargs.get("parttagger_weight", 1.0)
        self._parttaggers = [PartTagger(self, id="Part-{}".format(i+1), **self._args) for i in range(self._parttagger_num)]

        self._parsers = []

        self._hsel_weight = kwargs.get("hsel_weight", 1.0)
        self._hsel_parser = HSelParser(self, id="HSel-1", **self._args)
        self._parsers.append(self._hsel_parser)

        self._label_weight = kwargs.get("label_weight", 1.0)
        self._labeler = Labeler(self, **self._args)

        self._hselpartparser = HSelPartParser(self, self._hsel_parser, self._parttaggers[0], self._labeler)
        self._parsers.append(self._hselpartparser)

        return self

    def _load_vocab(self, vocab):
        self._fullvocab = vocab
        self._upos = {p: i for i, p in enumerate(vocab["upos"])}
        self._iupos = vocab["upos"]
        self._xpos = {p: i for i, p in enumerate(vocab["xpos"])}
        self._ixpos = vocab["xpos"]
        self._part = {p: i for i, p in enumerate(vocab["part"])}
        self._ipart = vocab["part"]
        self._layers = [{p: i for i, p in enumerate(layer)} for layer in vocab["layers"]]
        self._ilayers = vocab["layers"]
        self._vocab = {w: i + 3 for i, w in enumerate(vocab["vocab"])}
        self._wordfreq = vocab["wordfreq"]
        self._charset = {c: i + 3 for i, c in enumerate(vocab["charset"])}
        self._charfreq = vocab["charfreq"]
        self._rels = {r: i for i, r in enumerate(vocab["rels"])}
        self._irels = vocab["rels"]
        self._relcats = {r: i + 1 for i, r in enumerate(vocab["relcats"])}
        self._irelcats = vocab["relcats"]
        self._feats = {f: i + 1 for i, f in enumerate(vocab["feats"])}
        self._rel2cat = np.zeros(len(self._irels), dtype=int)
        self._core = vocab["core"]
        self._cat_layers = vocab["cat_layers"]
        self._label_mapping = vocab["label_mapping"]
        for i, r in enumerate(self._irels):
            if r in self._core:
                if r in self._label_mapping:
                    self._rel2cat[i] = self._relcats[self._label_mapping[r]]
                else:
                    self._rel2cat[i] = self._relcats[r]
            else:
                self._rel2cat[i] = self._relcats["_noncore"]
        self._partmappings, self._rpartmappings = build_part_mappings(self._ipart, self._relcats)
        print(len(self._partmappings), "Parts X Width")
        sys.stdout.flush()
        self._layermappings, self._rlayermappings = build_part_mappings([x for y in self._ilayers for x in y], self._relcats)
        print(len(self._layermappings), "Layers X Parts X Width")
        sys.stdout.flush()

    def load_vocab(self, filename):
        with open(filename, "r") as f:
            vocab = json.load(f)
        self._load_vocab(vocab)
        return self

    def save_vocab(self, filename):
        with open(filename, "w") as f:
            json.dump(self._fullvocab, f)
        return self

    def build_vocab(self, filename, savefile=None, cutoff=1, partcutoff=1):
        if isinstance(filename, str):
            graphs = read_conll(filename)
        elif isinstance(filename, list):
            graphs = []
            for f in filename:
                graphs.extend(read_conll(f))

        self._fullvocab= buildVocab(graphs, cutoff, partcutoff)

        if savefile:
            self.save_vocab(savefile)
        self._load_vocab(self._fullvocab)
        return self

    def save_model(self, filename):
        self.save_vocab(filename + ".vocab")
        with open(filename + ".params", "w") as f:
            json.dump(self._args, f)
        self._model.save(filename + ".model")
        return self

    def load_model(self, filename, **kwargs):
        self.load_vocab(filename + ".vocab")
        with open(filename + ".params", "r") as f:
            args = json.load(f)
            args.update(kwargs)
            self.create_parser(**args)
        self.init_model()
        self._model.populate(filename + ".model")
        return self


    def init_model(self):
        self._model = Model()
        if self._optimizer == "adam":
            self._trainer = AdamTrainer(self._model, alpha=self._learning_rate, beta_1 = self._beta1, beta_2=self._beta2, eps=self._epsilon)
        if self._optimizer == "amsgrad":
            self._trainer = AmsgradTrainer(self._model, alpha=self._learning_rate, beta_1 = self._beta1, beta_2=self._beta2, eps=self._epsilon)
        elif self._optimizer == "sgd":
            self._trainer = SimpleSGDTrainer(self._model, self._learning_rate)
        elif self._optimizer == "mom":
            self._trainer = MomentumSGDTrainer(self._model, self._learning_rate)

        self._trainer.set_sparse_updates(self._sparse_updates)
        self._trainer.set_clip_threshold(self._clip)

        input_dims = 0
        if self._cdims > 0 and self._char_lstm_dims > 0:
            if self._char_lstm_dims > 0:
                self._char_lookup = self._model.add_lookup_parameters((len(self._charset) + 3, self._cdims))
                self._char_lstm = BiRNNBuilder(self._char_lstm_layers, self._cdims, self._char_lstm_dims, self._model, VanillaLSTMBuilder)
                if self._char_repr_method == "concat":
                    input_dims += self._char_lstm_dims

            if self._char_repr_method == "pred":
                self._char_to_word = Dense(self._char_lstm_dims, self._wdims, tanh, self._model)

        if self._wdims > 0:
            self._word_lookup = self._model.add_lookup_parameters((len(self._vocab) + 3, self._wdims))
            input_dims += self._wdims

        if self._pdims > 0:
            self._upos_lookup = self._model.add_lookup_parameters((len(self._upos) + 1, self._pdims))
            input_dims += self._pdims

        if self._fdims > 0:
            self._feats_lookup = self._model.add_lookup_parameters((len(self._feats) + 1, self._fdims))
            input_dims += self._fdims

        if self._edims > 0:
            self._eoov = self._model.add_parameters(self._edims)
            input_dims += self._edims
            self._external_embeddings = []
            self._external_mappings = {}

        if input_dims <= 0:
            print("Input to LSTM is empty! You need to use at least one of word embeddings or character embeddings.")
            return

        if self._bidirectional:
            self._bilstm = BiRNNBuilder(self._bilstm_layers, input_dims, self._bilstm_dims, self._model, self.WordLSTMBuilder)
        else:
            self._bilstm = CoupledLSTMBuilder(self._bilstm_layers, input_dims, self._bilstm_dims, self._model)

        self._root_repr = self._model.add_parameters(input_dims)
        self._bos_repr = self._model.add_parameters(input_dims)
        self._eos_repr = self._model.add_parameters(input_dims)


        for utagger in self._utaggers:
            utagger.init_params()

        for xtagger in self._xtaggers:
            xtagger.init_params()

        for parttagger in self._parttaggers:
            parttagger.init_params()

        for parser in self._parsers:
            parser.init_params()

        self._labeler.init_params()

        self._model.set_weight_decay_lambda(self._weight_decay)

        return self

    def load_embeddings(self, filename):
        if not os.path.isfile(filename + ".vocab"):
            return self

        with open(filename + ".vocab", "r") as f:
            self._external_mappings = json.load(f)
        self._external_embeddings = np.load(filename + ".npy")
        print("Loaded embeddings from", filename, self._external_embeddings.shape)

        return self

    def _next_epoch(self):
        self._epoch += 1
        return self


    def _get_lstm_features(self, sentence, train=False, pre_lstm=False):
        carriers = [ComputationCarrier() for i in range(len(sentence))]
        carriers[0].vec = self._root_repr
        carriers[0].word_id = 0
        carriers[0].pos_id = 0

        for entry, cc in zip(sentence[1:], carriers[1:]):
            cc.word_id = self._vocab.get(entry.norm, 0)
            cc.pos_id = self._upos.get(entry.upos, len(self._upos))
            vecs = []

            word_flag = False
            if self._wdims > 0:
                c = float(self._wordfreq.get(entry.norm, 0))
                word_flag = c > 0 and (not train or (random.random() < (c / (self._word_smooth + c))))

                wvec = lookup(self._word_lookup, int(self._vocab.get(entry.norm, 0)) if word_flag else 0)
                if train and self._block_dropout > 0.:
                    wvec = block_dropout(wvec, self._block_dropout)
                if self._char_repr_method == "concat" or word_flag:
                    vecs.append(wvec)

            if self._cdims > 0 and self._char_lstm_dims > 0:
                if not (self._char_repr_method == "pred" and word_flag):
                    char_vecs = []
                    char_vecs.append(lookup(self._char_lookup, 1))
                    for ch in entry.word:
                        c = float(self._charfreq.get(ch, 0))
                        keep_flag = not train or (random.random() < (c / (self._char_smooth + c)))
                        charvec = lookup(self._char_lookup, int(self._charset.get(ch, 0)) if keep_flag else 0)
                        if train and self._char_dropout > 0.:
                            char_vecs.append(block_dropout(charvec, self._char_dropout))
                        else:
                            char_vecs.append(charvec)
                    char_vecs.append(lookup(self._char_lookup, 2))

                    char_vecs = self._char_lstm.add_inputs(char_vecs)

                    cvec = concatenate([char_vecs[0][-1].output(), char_vecs[-1][0].output()])
                    if self._char_repr_method != "concat" and not word_flag:
                        cvec = self._char_to_word(cvec)
                    if train and self._block_dropout > 0.:
                        cvec = block_dropout(cvec, self._block_dropout)

                    if self._char_repr_method == "concat" or not word_flag:
                        vecs.append(cvec)

            if self._pdims > 0:
                pvec = lookup(self._upos_lookup, int(self._upos.get(entry.upos, len(self._upos))))
                if train and self._block_dropout > 0.:
                    pvec = block_dropout(pvec, self._block_dropout)
                vecs.append(pvec)


            if self._fdims > 0:
                feats = []
                for f in entry.feats_set:
                    if f in self._feats and ((not train) or random.random() > self._feature_dropout):
                        feats.append(lookup(self._feats_lookup, int(self._feats[f])))

                if len(feats) == 0:
                    feats.append(lookup(self._feats_lookup, 0))

                vecs.append(emax(feats))

            if self._edims > 0:
                e = self._external_mappings.get(entry.norm, -1)
                if e >= 0:
                    evec = self._external_embeddings[e]
                    if len(evec) == self._edims:
                        evec = inputTensor(evec)
                    else:
                        evec = self._eoov
                else:
                    evec = self._eoov
                if train and self._block_dropout > 0.:
                    evec = block_dropout(evec, self._block_dropout)
                vecs.append(evec)

            cc.vec = concatenate(vecs)

        if pre_lstm:
            return carriers

        if self._bidirectional:
            ret = self._bilstm.transduce([x.vec for x in carriers[:]])
        else:
            ret = self._bilstm.initial_state().transduce([x.vec for x in carriers[:]])

        for vec, cc in zip(ret[:], carriers[:]):
            cc.vec = vec

        for entry, cc in zip(sentence, carriers):
            cc.node = entry

        return carriers


    def _minibatch_update(self, loss, num_tokens):
        if len(loss) == 0:
            self._init_cg(train=True)
            return 0.

        loss = esum(loss) * (1. / self._batch_size)
        ret = loss.scalar_value()
        loss.backward()
        self._trainer.update()
        self._steps += 1
        self._init_cg(train=True)

        return ret * self._batch_size


    def get_lstm_features(self, graph, pre_lstm=False):
        self._init_cg(train=False)
        carriers = self._get_lstm_features(graph.nodes, train=False, pre_lstm=pre_lstm)

        return [c.vec.npvalue() for c in carriers]


    def predict(self, graphs, **kwargs):
        label = kwargs.get("label", False)
        hsel = kwargs.get("hsel", False)
        hselpart = kwargs.get("hselpart", False)
        supertag = kwargs.get("supertag", False)
        self._hsel_steps = 0
        if hselpart: self._hselpartparser.failed = 0

        parsers = []
        if hsel: parsers.append(self._hsel_parser)

        for graph in tqdm(graphs):
            self._init_cg(train=False)
            carriers = self._get_lstm_features(graph.nodes, train=False)
            if hselpart:
                self._hselpartparser.predict(graph, carriers)
            else:
                for parser in parsers:
                    parser.predict(graph, carriers)

                if supertag:
                    for parttagger in self._parttaggers:
                        parttagger.predict(graph, carriers)
                        for x in graph.nodes:
                            x.xpos = x.part

                if label:
                    self._labeler.predict(graph, carriers)
        if hselpart: print("HselPart", self._hselpartparser.failed, "failed")

        return graphs


    def test(self, graphs=None, filename=None, **kwargs):
        utag = kwargs.get("utag", False)
        xtag = kwargs.get("xtag", False)
        part = kwargs.get("part", False)
        hsel = kwargs.get("hsel", False)
        label = kwargs.get("label", False)
        hselpart = kwargs.get("hselpart", False)
        save_prefix = kwargs.get("save_prefix", None)

        if graphs is None:
            graphs = read_conll(filename)

        total = 0
        correct_counts = defaultdict(int)
        label_correct = 0
        part_correct = 0
        ret = 0.

        parsers = []
        if hsel: parsers.append(self._hsel_parser)
        if hsel: self._hsel_steps = 0

        for gold_graph in graphs:
            self._init_cg(train=False)
            graph = gold_graph.cleaned(node_level=False)
            carriers = self._get_lstm_features(graph.nodes, train=False)

            total += len(carriers) - 1

            gold_upos = [x.upos for x in gold_graph.nodes]

            if utag:
                for utagger in self._utaggers:
                    utagger.predict(graph, carriers)
                    predicted = [x.upos for x in graph.nodes]
                    correct_counts["{} Accuracy".format(utagger.id)] += POSCorrect(predicted, gold_upos)

            if xtag:
                gold_xpos = [x.xpos for x in gold_graph.nodes]
                for xtagger in self._xtaggers:
                    xtagger.predict(graph, carriers)
                    predicted = [x.xpos for x in graph.nodes]
                    correct_counts["{} Accuracy".format(xtagger.id)] += POSCorrect(predicted, gold_xpos)

            if part:
                gold_layers = [x.layers for x in gold_graph.nodes]
                for parttagger in self._parttaggers:
                    parttagger.predict(graph, carriers)
                    for x in graph.nodes:
                        x.xpos = "|".join(x.layers)
                    for i in range(len(self._layers)):
                        gold = [x[i] for x in gold_layers]
                        predicted = [x.layers[i] for x in graph.nodes]
                        tmp = POSCorrect(predicted, gold) / len(self._layers)
                        correct_counts["{} Accuracy".format(parttagger.id)] += tmp
                        part_correct += tmp

                        c = 0
                        t = 0
                        for p, g, u in zip(predicted, gold, gold_upos):
                            if u == "VERB":
                                if p == g:
                                    c += 1
                                t += 1
                        correct_counts["{}-VERB Accuracy".format(parttagger.id)] += c / len(self._layers)
                        correct_counts["{}-VERB Total".format(parttagger.id)] += t / len(self._layers)

            for parser in parsers:
                parser.predict(graph, carriers)
                if label:
                    self._labeler.predict(graph, carriers)

                for i in range(1, len(carriers)):
                    if gold_graph.heads[i] == graph.heads[i]:
                        correct_counts["{}-UAS".format(parser.id)] += 1
                        if gold_graph.rels[i].split(":")[0] == graph.rels[i].split(":")[0]:
                            correct_counts["{}-LAS".format(parser.id)] += 1

            if hselpart:
                self._hselpartparser.predict(graph, carriers)

                for i in range(1, len(carriers)):
                    if gold_graph.heads[i] == graph.heads[i]:
                        correct_counts["{}-UAS".format(self._hselpartparser.id)] += 1
                        if gold_graph.rels[i].split(":")[0] == graph.rels[i].split(":")[0]:
                            correct_counts["{}-LAS".format(self._hselpartparser.id)] += 1

            if len(parsers) == 0 and label:
                graph.heads = np.copy(gold_graph.heads)
                self._labeler.predict(graph, carriers)
                for i in range(1, len(carriers)):
                    if gold_graph.rels[i].split(":")[0] == graph.rels[i].split(":")[0]:
                        label_correct += 1


        for id in sorted(correct_counts):
            print(id, correct_counts[id] / total)
            if label and "-LAS" in id:
                ret = max(ret, correct_counts[id])
            if not label and "-UAS" in id:
                ret = max(ret, correct_counts[id])

        if len(parsers) == 0:
            if label:
                print("LA", label_correct / total)
                ret = max(ret, label_correct)
            elif part:
                print("PA", part_correct / total)
                ret = max(ret, part_correct)

        if hsel:
            print("Average steps:", self._hsel_steps / total)
        sys.stdout.flush()

        return ret / total


    def fine_tune(self, filename, dev_portion=0.95, max_steps=1000, eval_steps=100, decay_evals=5, decay_times=0, dev=None, **kwargs):
        graphs = read_conll(filename)
        graphs_list = [list(random.sample(graphs, int(len(graphs) * dev_portion)))]

        return self.train("", max_steps, eval_steps, decay_evals, decay_times, dev, graphs_list, **kwargs)

    def train_small(self, filename, split_ratio=0.9, **kwargs):
        save_prefix = kwargs.get("save_prefix", None)
        graphs = read_conll(filename)
        random.shuffle(graphs)
        train_len = int(len(graphs) * split_ratio)
        train_graphs = [graphs[:train_len]]
        dev_graphs = graphs[train_len:]

        if save_prefix is not None:
            write_conll("{}_train.conllu".format(save_prefix), train_graphs[0])
            write_conll("{}_dev.conllu".format(save_prefix), dev_graphs)

        return self.train("", graphs_list=train_graphs, dev_graphs=dev_graphs, **kwargs)


    def train(self, filename, max_steps=1000, eval_steps=100, decay_evals=5, decay_times=0, decay_ratio=0.5, dev=None, graphs_list=None, dev_portion=0.8, dev_graphs=None, **kwargs):
        if graphs_list is None:
            if isinstance(filename, str):
                graphs_list = [read_conll(filename)]
            elif isinstance(filename, list):
                graphs_list = [read_conll(f) for f in filename]

        total = 0
        proj_count = 0
        total_trees = 0
        for graphs in graphs_list:
            total_trees += len(graphs)
            for g in graphs:
                total += len(g.heads) - 1
                if is_projective(g.heads):
                    g.proj_heads = g.heads
                    proj_count += 1
                else:
                    g.proj_heads = projectivize(g.heads)
        print("Training set projective ratio", proj_count / total_trees)
        sys.stdout.flush()

        train_set_steps = total / self._batch_size

        eval_steps = max(int(train_set_steps * 0.25), eval_steps)

        save_prefix = kwargs.get("save_prefix", None)

        if dev is not None and dev_graphs is None:
            dev_graphs = read_conll(dev)
            dev_samples = int(len(dev_graphs) * dev_portion)
            dev_graphs = list(random.sample(dev_graphs, dev_samples))
            if save_prefix is not None:
                write_conll("{}_dev.conllu".format(save_prefix), dev_graphs)


        utag = kwargs.get("utag", False)
        xtag = kwargs.get("xtag", False)
        part = kwargs.get("part", False)
        hsel = kwargs.get("hsel", False)
        label = kwargs.get("label", False)

        self._steps = 0
        self._epoch = 0
        self._base_lr = 1.
        max_dev = 0.
        max_dev_ep = 0

        i = 0
        t0 = time.time()

        self._init_cg(train=True)
        loss = []
        loss_sum = 0.0
        total_tokens = 0
        num_tokens = 0
        correct_counts = defaultdict(float)

        label_correct = 0

        for graph in shuffled_balanced_stream(graphs_list):
            i += 1
            if i % 100 == 0:
                print(i, "{0:.2f}s".format(time.time() - t0), end=" ")
                sys.stdout.flush()
                t0 = time.time()

            carriers = self._get_lstm_features(graph.nodes, train=True)

            num_tokens += len(carriers) - 1
            total_tokens += len(carriers) - 1

            if utag:
                for utagger in self._utaggers:
                    c, l = utagger.sent_loss(graph, carriers)
                    correct_counts[utagger.id] += c
                    if len(l) > 0:
                        loss.append(esum(l) * (self._utagger_weight / self._utagger_num))

            if xtag:
                for xtagger in self._xtaggers:
                    c, l = xtagger.sent_loss(graph, carriers)
                    correct_counts[xtagger.id] += c
                    if len(l) > 0:
                        loss.append(esum(l) * (self._xtagger_weight / self._xtagger_num))

            if part:
                for parttagger in self._parttaggers:
                    c, l = parttagger.sent_loss(graph, carriers)
                    correct_counts[parttagger.id] += c
                    if len(l) > 0:
                        loss.append(esum(l) * (self._parttagger_weight / self._parttagger_num))

            if hsel:
                c, l = self._hsel_parser.sent_loss(graph, carriers)
                correct_counts[self._hsel_parser.id] += c
                if len(l) > 0:
                    loss.append(esum(l) * self._hsel_weight)

            if label:
                c, l = self._labeler.sent_loss(graph, carriers)
                label_correct += c
                if len(l) > 0:
                    loss.append(esum(l) * self._label_weight)

            if num_tokens >= self._batch_size:
                loss_sum += self._minibatch_update(loss, num_tokens)
                loss = []
                num_tokens = 0

                if self._steps % eval_steps == 0:

                    self._next_epoch()
                    print()
                    self._trainer.status()

                    print()
                    print("Total Loss", loss_sum, "Avg", loss_sum / total_tokens)
                    for id in sorted(correct_counts):
                        print("Train {} Acc".format(id), correct_counts[id] / total_tokens)
                    if label:
                        print("Train Label Acc", label_correct / total_tokens)
                    sys.stdout.flush()

                    loss_sum = 0.0
                    total_tokens = 0
                    num_tokens = 0
                    correct_counts = defaultdict(float)
                    label_correct = 0

                    if self._steps >= max_steps:
                        break

                    if dev_graphs is not None:
                        performance = self.test(graphs=dev_graphs, **kwargs)
                        self._init_cg(train=True)

                        if performance >= max_dev:
                            max_dev = performance
                            max_dev_ep = 0
                            if save_prefix:
                                self.save_model("{}_{}_model".format(save_prefix, 0))
                        else:
                            max_dev_ep += 1

                        if max_dev_ep >= decay_evals:
                            if decay_times > 0:
                                decay_times -= 1
                                max_dev_ep = 0
                                self._base_lr *= decay_ratio
                                self._trainer.restart(self._learning_rate * self._base_lr)
                                print("Learning rate decayed!")
                                print("Current decay ratio", self._base_lr * math.pow(self._anneal_base, self._steps / self._anneal_steps))
                                sys.stdout.flush()
                            else:
                                break

        return self

    def _init_cg(self, train=False):
        renew_cg()
        if train:
            self._bilstm.set_dropout(self._bilstm_dropout)
            if self._cdims > 0 and self._char_lstm_dims > 0:
                self._char_lstm.set_dropout(self._char_lstm_dropout)
        else:
            self._bilstm.set_dropout(0.)
            if self._cdims > 0 and self._char_lstm_dims > 0:
                self._char_lstm.set_dropout(0.)
        for utagger in self._utaggers:
            utagger.init_cg(train)
        for xtagger in self._xtaggers:
            xtagger.init_cg(train)
        for parttagger in self._parttaggers:
            parttagger.init_cg(train)

        for parser in self._parsers:
            parser.init_cg(train)

    def finish(self, **kwargs):
        print()
        sys.stdout.flush()


if __name__ == '__main__':
    fire.Fire(CDParser)
