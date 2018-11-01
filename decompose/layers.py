#!/usr/bin/env python
# encoding: utf-8


from dynet import parameter, transpose, dropout, rectify
from dynet import layer_norm, affine_transform
from dynet import concatenate, zeroes, dot_product
from dynet import GlorotInitializer, ConstInitializer, SaxeInitializer, NumpyInitializer
from dynet import BiRNNBuilder
from dynet import logistic, cmult, dropout_dim, pick
from dynet import random_bernoulli, tanh
from dynet import bmax
import math
import numpy as np


class Dense(object):
    def __init__(self, indim, outdim, activation, model, ln=False):
        self.model = model
        self.activation = activation
        self.ln = ln
        if activation == rectify:
            self.W = model.add_parameters((outdim, indim), init=GlorotInitializer(gain=math.sqrt(2.)))
        else:
            self.W = model.add_parameters((outdim, indim))
        self.b = model.add_parameters(outdim, init=ConstInitializer(0.))
        if ln:
            self.ln_s = model.add_parameters(outdim, ConstInitializer(1.))
        self.spec = (indim, outdim, activation, ln)

    def __call__(self, x):
        if self.ln:
            #  return self.activation(layer_norm(parameter(self.W) * x, parameter(self.ln_s), parameter(self.b)))
            return self.activation(layer_norm(self.W * x, self.ln_s, self.b))
        else:
            return self.activation(affine_transform([self.b, self.W, x]))

    def param_collection(self):
        return self.model

    @staticmethod
    def from_spec(spec, model):
        (indim, outdim, activation, ln) = spec
        return Dense(indim, outdim, activation, model, ln)


class MultiLayerPerceptron(object):
    def __init__(self, dims, activation, model, ln=False):
        self.model = model
        self.layers = []
        self.dropout = 0.
        self.dropout_dim = -1
        self.outdim = []
        for indim, outdim in zip(dims, dims[1:]):
            self.layers.append(Dense(indim, outdim, activation, model, ln))
            self.outdim.append(outdim)
        self.spec = (indim, outdim, activation, ln)

    def __call__(self, x):
        for layer, dim in zip(self.layers, self.outdim):
            x = layer(x)
            if self.dropout > 0.:
                if self.dropout_dim >= 0:
                    x = dropout_dim(x, self.dropout_dim, self.dropout)
                else:
                    x = dropout(x, self.dropout)
        return x

    def set_dropout(self, droprate, dim=-1):
        self.dropout = droprate
        self.dropout_dim = dim

    def param_collection(self):
        return self.model

    @staticmethod
    def from_spec(spec, model):
        (indim, outdim, activation, ln) = spec
        return MultiLayerPerceptron(indim, outdim, activation, model, ln)


class Bilinear(object):
    def __init__(self, dim, model):
        self.U = model.add_parameters((dim, dim), init=SaxeInitializer())

    def __call__(self, x, y):
        #  U = parameter(self.U)
        #  return transpose(x) * U * y
        return transpose(x) * self.U * y

    def get_components(self):
        return [self.U]

    def restore_components(self, components):
        [self.U] = components


class Biaffine(object):
    def __init__(self, indim, model):
        self.model = model
        self.U = Bilinear(indim, model)
        self.x_bias = model.add_parameters((indim))
        self.y_bias = model.add_parameters((indim))
        self.bias = model.add_parameters(1)
        self.spec = (indim,)

    def __call__(self, x, y):
        #  x_bias = parameter(self.x_bias)
        #  y_bias = parameter(self.y_bias)
        #  bias = parameter(self.bias)

        #  return bias + dot_product(x_bias, x) + dot_product(y_bias, y) + self.U(x, y)
        return self.bias + dot_product(self.x_bias, x) + dot_product(self.y_bias, y) + self.U(x, y)

    def param_collection(self):
        return self.model

    @staticmethod
    def from_spec(spec, model):
        (indim, ) = spec
        return Biaffine(indim, model)


class BiaffineBatch(object):
    def __init__(self, indim, outdim, model):
        self.model = model
        self.U = [Bilinear(indim + 1, model) for i in range(outdim)]
        self.spec = (indim, outdim)

    def __call__(self, x, y):
        x = concatenate([x, zeroes((1, x.dim()[0][1],)) + 1.])
        y = concatenate([y, zeroes((1, y.dim()[0][1],)) + 1.])

        if self.spec[1] == 1:
            return self.U[0](x, y)
        else:
            return concatenate([u(x, y) for u in self.U], 2)

    def get_components(self):
        return self.U

    def restore_components(self, components):
        self.U = components[:-3]

    def param_collection(self):
        return self.model

    @staticmethod
    def from_spec(spec, model):
        (indim, outdim) = spec
        return BiaffineBatch(indim, outdim, model)


class CoupledLSTM:

    def __init__(self, layers, input_dim, hidden_dim, model):
        layers, input_dim, hidden_dim = int(layers), int(input_dim), int(hidden_dim)
        self._spec = (layers, input_dim, hidden_dim)
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        model = self.model = model.add_subcollection("coupledlstm")
        self.wix = model.add_parameters((hidden_dim, input_dim))
        self.wih = model.add_parameters((hidden_dim, hidden_dim))
        self.wcx = model.add_parameters((hidden_dim, input_dim))
        self.wch = model.add_parameters((hidden_dim, hidden_dim))
        self.wox = model.add_parameters((hidden_dim, input_dim))
        self.woh = model.add_parameters((hidden_dim, hidden_dim))
        self.bi = model.add_parameters(hidden_dim, init=ConstInitializer(0.))
        self.bc = model.add_parameters(hidden_dim, init=ConstInitializer(0.))
        self.bo = model.add_parameters(hidden_dim, init=ConstInitializer(0.))
        self.initc = model.add_parameters(hidden_dim)
        self.inith = model.add_parameters(hidden_dim)
        self.set_dropout(0.)

    def param_collection(self):
        return self.model

    @property
    def spec(self):
        return self._spec

    @classmethod
    def from_spec(cls, spec, model):
        return CoupledLSTM(*spec, model)

    def transduce(self, es):
        ret = []
        #  wix, wih, bi = parameter(self.wix), parameter(self.wih), parameter(self.bi)
        #  wcx, wch, bc = parameter(self.wcx), parameter(self.wch), parameter(self.bc)
        #  wox, woh, bo = parameter(self.wox), parameter(self.woh), parameter(self.bo)
        #  prev_c, prev_h = parameter(self.initc), parameter(self.inith)
        wix, wih, bi = self.wix, self.wih, self.bi
        wcx, wch, bc = self.wcx, self.wch, self.bc
        wox, woh, bo = self.wox, self.woh, self.bo
        prev_c, prev_h = self.initc, self.inith

        if self.dropout_x > 0.:
            retention_x = 1. - self.dropout_x
            scale_x = 1. / retention_x
            mask_x_i = random_bernoulli(self._input_dim, p=retention_x, scale=scale_x)
            mask_x_c = random_bernoulli(self._input_dim, p=retention_x, scale=scale_x)
            mask_x_o = random_bernoulli(self._input_dim, p=retention_x, scale=scale_x)
        if self.dropout_h > 0.:
            retention_h = 1. - self.dropout_h
            scale_h = 1. / retention_h
            mask_h_i = random_bernoulli(self._hidden_dim, p=retention_h, scale=scale_h)
            mask_h_c = random_bernoulli(self._hidden_dim, p=retention_h, scale=scale_h)
            mask_h_o = random_bernoulli(self._hidden_dim, p=retention_h, scale=scale_h)

        for x in es:
            ait = affine_transform([bi, wix, cmult(mask_x_i, x) if self.dropout_x > 0. else x, wih, cmult(mask_h_i, prev_h) if self.dropout_h > 0. else prev_h])
            it = logistic(ait)
            ft = 1. - it

            atct = affine_transform([bc, wcx, cmult(mask_x_c, x) if self.dropout_x > 0. else x, wch, cmult(mask_h_c, prev_h) if self.dropout_h > 0. else prev_h])
            tct = tanh(atct)
            ct = prev_c + cmult(tct - prev_c, it)

            aot = affine_transform([bo, wox, cmult(mask_x_o, x) if self.dropout_x > 0. else x, woh, cmult(mask_h_o, prev_h) if self.dropout_h > 0. else prev_h])
            ot = logistic(aot)
            h = cmult(tanh(ct), ot)

            ret.append(h)
            prev_c = ct
            prev_h = h
        return ret

    def set_dropout(self, dropout):
        self.dropout_x = self.dropout_h = dropout

    def disable_dropout(self, dropout):
        self.dropout_x = self.dropout_h = 0.

    def initial_state(self):
        return self


class CoupledHighwayLSTM:

    def __init__(self, layers, input_dim, hidden_dim, model):
        layers, input_dim, hidden_dim = int(layers), int(input_dim), int(hidden_dim)
        self._spec = (layers, input_dim, hidden_dim)
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        model = self.model = model.add_subcollection("coupledhighwaylstm")
        self.wix = model.add_parameters((hidden_dim, input_dim))
        self.wih = model.add_parameters((hidden_dim, hidden_dim))
        self.wcx = model.add_parameters((hidden_dim, input_dim))
        self.wch = model.add_parameters((hidden_dim, hidden_dim))
        self.wox = model.add_parameters((hidden_dim, input_dim))
        self.woh = model.add_parameters((hidden_dim, hidden_dim))
        self.wrx = model.add_parameters((hidden_dim, input_dim))
        self.wrh = model.add_parameters((hidden_dim, hidden_dim))
        self.whx = model.add_parameters((hidden_dim, input_dim))
        self.bi = model.add_parameters(hidden_dim, init=ConstInitializer(0.))
        self.bc = model.add_parameters(hidden_dim, init=ConstInitializer(0.))
        self.bo = model.add_parameters(hidden_dim, init=ConstInitializer(0.))
        self.br = model.add_parameters(hidden_dim, init=ConstInitializer(0.))
        self.initc = model.add_parameters(hidden_dim)
        self.inith = model.add_parameters(hidden_dim)
        self.set_dropout(0.)

    def param_collection(self):
        return self.model

    @property
    def spec(self):
        return self._spec

    @classmethod
    def from_spec(cls, spec, model):
        return CoupledHighwayLSTM(*spec, model)

    def transduce(self, es):
        ret = []
        #  wix, wih, bi = parameter(self.wix), parameter(self.wih), parameter(self.bi)
        #  wcx, wch, bc = parameter(self.wcx), parameter(self.wch), parameter(self.bc)
        #  wox, woh, bo = parameter(self.wox), parameter(self.woh), parameter(self.bo)
        #  wrx, wrh, whx, br = parameter(self.wrx), parameter(self.wrh), parameter(self.whx), parameter(self.br)
        #  prev_c, prev_h = parameter(self.initc), parameter(self.inith)
        wix, wih, bi = self.wix, self.wih, self.bi
        wcx, wch, bc = self.wcx, self.wch, self.bc
        wox, woh, bo = self.wox, self.woh, self.bo
        wrx, wrh, whx, br = self.wrx, self.wrh, self.whx, self.br
        prev_c, prev_h = self.initc, self.inith

        if self.dropout_x > 0.:
            retention_x = 1. - self.dropout_x
            scale_x = 1. / retention_x
            mask_x_i = random_bernoulli(self._input_dim, p=retention_x, scale=scale_x)
            mask_x_c = random_bernoulli(self._input_dim, p=retention_x, scale=scale_x)
            mask_x_o = random_bernoulli(self._input_dim, p=retention_x, scale=scale_x)
            mask_x_r = random_bernoulli(self._input_dim, p=retention_x, scale=scale_x)
        if self.dropout_h > 0.:
            retention_h = 1. - self.dropout_h
            scale_h = 1. / retention_h
            mask_h_i = random_bernoulli(self._hidden_dim, p=retention_h, scale=scale_h)
            mask_h_c = random_bernoulli(self._hidden_dim, p=retention_h, scale=scale_h)
            mask_h_o = random_bernoulli(self._hidden_dim, p=retention_h, scale=scale_h)
            mask_h_r = random_bernoulli(self._hidden_dim, p=retention_h, scale=scale_h)

        for x in es:
            ait = affine_transform([bi, wix, cmult(mask_x_i, x) if self.dropout_x > 0. else x, wih, cmult(mask_h_i, prev_h) if self.dropout_h > 0. else prev_h])
            it = logistic(ait)

            ft = 1. - it

            atct = affine_transform([bc, wcx, cmult(mask_x_c, x) if self.dropout_x > 0. else x, wch, cmult(mask_h_c, prev_h) if self.dropout_h > 0. else prev_h])
            tct = tanh(atct)
            ct = prev_c + cmult(tct - prev_c, it)

            aot = affine_transform([bo, wox, cmult(mask_x_o, x) if self.dropout_x > 0. else x, woh, cmult(mask_h_o, prev_h) if self.dropout_h > 0. else prev_h])
            ot = logistic(aot)

            h = cmult(tanh(ct), ot)

            art = affine_transform([br, wrx, cmult(mask_x_r, x) if self.dropout_x > 0. else x, wrh, cmult(mask_h_r, prev_h) if self.dropout_h > 0. else prev_h])
            rt = logistic(art)
            highway_h = cmult(rt, h) + cmult(1. - rt, whx * x)

            ret.append(highway_h)
            prev_c = ct
            prev_h = highway_h

        return ret

    def set_dropout(self, dropout):
        self.dropout_x = self.dropout_h = dropout

    def disable_dropout(self, dropout):
        self.dropout_x = self.dropout_h = 0.

    def initial_state(self):
        return self


class LSTM:

    def __init__(self, layers, input_dim, hidden_dim, model):
        layers, input_dim, hidden_dim = int(layers), int(input_dim), int(hidden_dim)
        self._spec = (layers, input_dim, hidden_dim)
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        model = self.model = model.add_subcollection("lstm")
        self.wix = model.add_parameters((hidden_dim, input_dim))
        self.wih = model.add_parameters((hidden_dim, hidden_dim))
        self.wfx = model.add_parameters((hidden_dim, input_dim))
        self.wfh = model.add_parameters((hidden_dim, hidden_dim))
        self.wcx = model.add_parameters((hidden_dim, input_dim))
        self.wch = model.add_parameters((hidden_dim, hidden_dim))
        self.wox = model.add_parameters((hidden_dim, input_dim))
        self.woh = model.add_parameters((hidden_dim, hidden_dim))
        self.bi = model.add_parameters(hidden_dim, init=ConstInitializer(0.))
        self.bf = model.add_parameters(hidden_dim, init=ConstInitializer(0.))
        self.bc = model.add_parameters(hidden_dim, init=ConstInitializer(0.))
        self.bo = model.add_parameters(hidden_dim, init=ConstInitializer(0.))
        self.initc = model.add_parameters(hidden_dim)
        self.inith = model.add_parameters(hidden_dim)
        self.set_dropout(0.)

    def param_collection(self):
        return self.model

    @property
    def spec(self):
        return self._spec

    @classmethod
    def from_spec(cls, spec, model):
        return LSTM(*spec, model)

    def transduce(self, es):
        ret = []
        #  wix, wih, bi = parameter(self.wix), parameter(self.wih), parameter(self.bi)
        #  wfx, wfh, bf = parameter(self.wfx), parameter(self.wfh), parameter(self.bf)
        #  wcx, wch, bc = parameter(self.wcx), parameter(self.wch), parameter(self.bc)
        #  wox, woh, bo = parameter(self.wox), parameter(self.woh), parameter(self.bo)
        #  prev_c, prev_h = parameter(self.initc), parameter(self.inith)
        wix, wih, bi = self.wix, self.wih, self.bi
        wfx, wfh, bf = self.wfx, self.wfh, self.bf
        wcx, wch, bc = self.wcx, self.wch, self.bc
        wox, woh, bo = self.wox, self.woh, self.bo
        prev_c, prev_h = self.initc, self.inith

        if self.dropout_x > 0.:
            retention_x = 1. - self.dropout_x
            scale_x = 1. / retention_x
            mask_x_i = random_bernoulli(self._input_dim, p=retention_x, scale=scale_x)
            mask_x_f = random_bernoulli(self._input_dim, p=retention_x, scale=scale_x)
            mask_x_c = random_bernoulli(self._input_dim, p=retention_x, scale=scale_x)
            mask_x_o = random_bernoulli(self._input_dim, p=retention_x, scale=scale_x)
        if self.dropout_h > 0.:
            retention_h = 1. - self.dropout_h
            scale_h = 1. / retention_h
            mask_h_i = random_bernoulli(self._hidden_dim, p=retention_h, scale=scale_h)
            mask_h_f = random_bernoulli(self._hidden_dim, p=retention_h, scale=scale_h)
            mask_h_c = random_bernoulli(self._hidden_dim, p=retention_h, scale=scale_h)
            mask_h_o = random_bernoulli(self._hidden_dim, p=retention_h, scale=scale_h)

        for x in es:
            ait = affine_transform([bi, wix, cmult(mask_x_i, x) if self.dropout_x > 0. else x, wih, cmult(mask_h_i, prev_h) if self.dropout_h > 0. else prev_h])
            it = logistic(ait)

            aft = affine_transform([bf, wfx, cmult(mask_x_f, x) if self.dropout_x > 0. else x, wfh, cmult(mask_h_f, prev_h) if self.dropout_h > 0. else prev_h])
            ft = logistic(aft)

            atct = affine_transform([bc, wcx, cmult(mask_x_c, x) if self.dropout_x > 0. else x, wch, cmult(mask_h_c, prev_h) if self.dropout_h > 0. else prev_h])
            tct = tanh(atct)

            ct = cmult(ft, prev_c) + cmult(it, tct)
            #  ct = prev_c + cmult(tct - prev_c, it)

            aot = affine_transform([bo, wox, cmult(mask_x_o, x) if self.dropout_x > 0. else x, woh, cmult(mask_h_o, prev_h) if self.dropout_h > 0. else prev_h])
            ot = logistic(aot)
            h = cmult(tanh(ct), ot)

            ret.append(h)
            prev_c = ct
            prev_h = h
        return ret

    def set_dropout(self, dropout):
        self.dropout_x = self.dropout_h = dropout

    def disable_dropout(self, dropout):
        self.dropout_x = self.dropout_h = 0.

    def initial_state(self):
        return self


class HighwayLSTM:

    def __init__(self, layers, input_dim, hidden_dim, model):
        layers, input_dim, hidden_dim = int(layers), int(input_dim), int(hidden_dim)
        self._spec = (layers, input_dim, hidden_dim)
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        model = self.model = model.add_subcollection("highwaylstm")
        self.wix = model.add_parameters((hidden_dim, input_dim))
        self.wih = model.add_parameters((hidden_dim, hidden_dim))
        self.wfx = model.add_parameters((hidden_dim, input_dim))
        self.wfh = model.add_parameters((hidden_dim, hidden_dim))
        self.wcx = model.add_parameters((hidden_dim, input_dim))
        self.wch = model.add_parameters((hidden_dim, hidden_dim))
        self.wox = model.add_parameters((hidden_dim, input_dim))
        self.woh = model.add_parameters((hidden_dim, hidden_dim))
        self.wrx = model.add_parameters((hidden_dim, input_dim))
        self.wrh = model.add_parameters((hidden_dim, hidden_dim))
        self.whx = model.add_parameters((hidden_dim, input_dim))
        self.bi = model.add_parameters(hidden_dim, init=ConstInitializer(0.))
        self.bf = model.add_parameters(hidden_dim, init=ConstInitializer(0.))
        self.bc = model.add_parameters(hidden_dim, init=ConstInitializer(0.))
        self.bo = model.add_parameters(hidden_dim, init=ConstInitializer(0.))
        self.br = model.add_parameters(hidden_dim, init=ConstInitializer(0.))
        self.initc = model.add_parameters(hidden_dim)
        self.inith = model.add_parameters(hidden_dim)
        self.set_dropout(0.)

    def param_collection(self):
        return self.model

    @property
    def spec(self):
        return self._spec

    @classmethod
    def from_spec(cls, spec, model):
        return HighwayLSTM(*spec, model)

    def transduce(self, es):
        ret = []
        #  wix, wih, bi = parameter(self.wix), parameter(self.wih), parameter(self.bi)
        #  wfx, wfh, bf = parameter(self.wfx), parameter(self.wfh), parameter(self.bf)
        #  wcx, wch, bc = parameter(self.wcx), parameter(self.wch), parameter(self.bc)
        #  wox, woh, bo = parameter(self.wox), parameter(self.woh), parameter(self.bo)
        #  wrx, wrh, whx, br = parameter(self.wrx), parameter(self.wrh), parameter(self.whx), parameter(self.br)
        #  prev_c, prev_h = parameter(self.initc), parameter(self.inith)
        wix, wih, bi = self.wix, self.wih, self.bi
        wfx, wfh, bf = self.wfx, self.wfh, self.bf
        wcx, wch, bc = self.wcx, self.wch, self.bc
        wox, woh, bo = self.wox, self.woh, self.bo
        wrx, wrh, whx, br = self.wrx, self.wrh, self.whx, self.br
        prev_c, prev_h = self.initc, self.inith

        if self.dropout_x > 0.:
            retention_x = 1. - self.dropout_x
            scale_x = 1. / retention_x
            mask_x_i = random_bernoulli(self._input_dim, p=retention_x, scale=scale_x)
            mask_x_f = random_bernoulli(self._input_dim, p=retention_x, scale=scale_x)
            mask_x_c = random_bernoulli(self._input_dim, p=retention_x, scale=scale_x)
            mask_x_o = random_bernoulli(self._input_dim, p=retention_x, scale=scale_x)
            mask_x_r = random_bernoulli(self._input_dim, p=retention_x, scale=scale_x)
        if self.dropout_h > 0.:
            retention_h = 1. - self.dropout_h
            scale_h = 1. / retention_h
            mask_h_i = random_bernoulli(self._hidden_dim, p=retention_h, scale=scale_h)
            mask_h_f = random_bernoulli(self._hidden_dim, p=retention_h, scale=scale_h)
            mask_h_c = random_bernoulli(self._hidden_dim, p=retention_h, scale=scale_h)
            mask_h_o = random_bernoulli(self._hidden_dim, p=retention_h, scale=scale_h)
            mask_h_r = random_bernoulli(self._hidden_dim, p=retention_h, scale=scale_h)

        for x in es:
            ait = affine_transform([bi, wix, cmult(mask_x_i, x) if self.dropout_x > 0. else x, wih, cmult(mask_h_i, prev_h) if self.dropout_h > 0. else prev_h])
            it = logistic(ait)

            aft = affine_transform([bf, wfx, cmult(mask_x_f, x) if self.dropout_x > 0. else x, wfh, cmult(mask_h_f, prev_h) if self.dropout_h > 0. else prev_h])
            ft = logistic(aft)

            atct = affine_transform([bc, wcx, cmult(mask_x_c, x) if self.dropout_x > 0. else x, wch, cmult(mask_h_c, prev_h) if self.dropout_h > 0. else prev_h])
            tct = tanh(atct)
            #  ct = prev_c + cmult(tct - prev_c, it)
            ct = cmult(ft, prev_c) + cmult(it, tct)

            aot = affine_transform([bo, wox, cmult(mask_x_o, x) if self.dropout_x > 0. else x, woh, cmult(mask_h_o, prev_h) if self.dropout_h > 0. else prev_h])
            ot = logistic(aot)

            h = cmult(tanh(ct), ot)

            art = affine_transform([br, wrx, cmult(mask_x_r, x) if self.dropout_x > 0. else x, wrh, cmult(mask_h_r, prev_h) if self.dropout_h > 0. else prev_h])
            rt = logistic(art)
            highway_h = cmult(rt, h) + cmult(1. - rt, whx * x)

            ret.append(highway_h)
            prev_c = ct
            prev_h = highway_h

        return ret

    def set_dropout(self, dropout):
        self.dropout_x = self.dropout_h = dropout

    def disable_dropout(self, dropout):
        self.dropout_x = self.dropout_h = 0.

    def initial_state(self):
        return self


def identity(x):
    return x

def leaky_relu(x):
    return bmax(.1 * x, x)
