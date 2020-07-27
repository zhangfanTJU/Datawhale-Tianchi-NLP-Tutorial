import math

import numpy as np
import torch
import torch.nn as nn


def orthonormal_initializer(output_size, input_size):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
    """
    # print(output_size, input_size)
    I = np.eye(output_size)
    lr = .1
    eps = .05 / (output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI ** 2 / 2)
            Q2 = Q ** 2
            Q -= lr * Q.dot(QTQmI) / (
                    np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        pass
        # print('Orthogonal pretrainer loss: %.2e' % loss)
    else:
        # print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class NoLinear(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, activation=None):
        super(NoLinear, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.linear = nn.Linear(in_features=input_size, out_features=hidden_size, bias=bias)
        if activation is None:
            self._activate = lambda x: x
        else:
            if not callable(activation):
                raise ValueError("activation must be callable: type={}".format(type(activation)))
            self._activate = activation

        self.reset_parameters()

    def forward(self, x):
        y = self.linear(x)
        return self._activate(y)

    def reset_parameters(self):
        W = orthonormal_initializer(self.hidden_size, self.input_size)
        self.linear.weight.data.copy_(torch.from_numpy(W))

        if self.bias:
            b = np.zeros(self.hidden_size, dtype=np.float32)
            self.linear.bias.data.copy_(torch.from_numpy(b))


def drop_input_independent(word_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.new_full((batch_size, seq_length), 1 - dropout_emb)
    word_masks = torch.bernoulli(word_masks)
    scale = 1.0 / (1.0 * word_masks + 1e-12)
    word_masks *= scale
    word_masks = word_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks

    return word_embeddings


def drop_sequence_sharedmask(inputs, dropout, batch_first=True):
    if batch_first:
        inputs = inputs.transpose(0, 1)
    seq_length, batch_size, hidden_size = inputs.size()
    drop_masks = inputs.data.new(batch_size, hidden_size).fill_(1 - dropout)
    drop_masks = torch.bernoulli(drop_masks)
    drop_masks = drop_masks / (1 - dropout)
    drop_masks = torch.unsqueeze(drop_masks, dim=2).expand(-1, -1, seq_length).permute(2, 0, 1)
    inputs = inputs * drop_masks

    return inputs.transpose(1, 0)


class LSTM(nn.Module):
    """A module that runs multiple steps of LSTM."""

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, \
                 bidirectional=False, dropout_in=0, dropout_out=0):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.num_directions = 2 if bidirectional else 1

        self.fcells = []
        self.bcells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            self.fcells.append(nn.LSTMCell(input_size=layer_input_size, hidden_size=hidden_size))
            if self.bidirectional:
                self.bcells.append(nn.LSTMCell(input_size=layer_input_size, hidden_size=hidden_size))

        self._all_weights = []
        for layer in range(num_layers):
            layer_params = (self.fcells[layer].weight_ih, self.fcells[layer].weight_hh, \
                            self.fcells[layer].bias_ih, self.fcells[layer].bias_hh)
            suffix = ''
            param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
            param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
            param_names = [x.format(layer, suffix) for x in param_names]
            for name, param in zip(param_names, layer_params):
                setattr(self, name, param)
            self._all_weights.append(param_names)

            if self.bidirectional:
                layer_params = (self.bcells[layer].weight_ih, self.bcells[layer].weight_hh, \
                                self.bcells[layer].bias_ih, self.bcells[layer].bias_hh)
                suffix = '_reverse'
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]
                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in range(self.num_layers):
            layer_input_size = self.input_size if layer == 0 else self.hidden_size * self.num_directions
            param_ih_name = 'weight_ih_l{}{}'.format(layer, '')
            param_hh_name = 'weight_hh_l{}{}'.format(layer, '')
            param_ih = self.__getattr__(param_ih_name)
            param_hh = self.__getattr__(param_hh_name)
            W = orthonormal_initializer(self.hidden_size, self.hidden_size + layer_input_size)
            W_h, W_x = W[:, :self.hidden_size], W[:, self.hidden_size:]
            param_ih.data.copy_(torch.from_numpy(np.concatenate([W_x] * 4, 0)))
            param_hh.data.copy_(torch.from_numpy(np.concatenate([W_h] * 4, 0)))

            if self.bidirectional:
                param_ih_name = 'weight_ih_l{}{}'.format(layer, '_reverse')
                param_hh_name = 'weight_hh_l{}{}'.format(layer, '_reverse')
                param_ih = self.__getattr__(param_ih_name)
                param_hh = self.__getattr__(param_hh_name)
                W = orthonormal_initializer(self.hidden_size, self.hidden_size + layer_input_size)
                W_h, W_x = W[:, :self.hidden_size], W[:, self.hidden_size:]
                param_ih.data.copy_(torch.from_numpy(np.concatenate([W_x] * 4, 0)))
                param_hh.data.copy_(torch.from_numpy(np.concatenate([W_h] * 4, 0)))

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(self.__getattr__(name), 0)

    @staticmethod
    def _forward_rnn(cell, input, masks, initial, drop_masks):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in range(max_time):
            h_next, c_next = cell(input=input[time], hx=hx)
            h_next = h_next * masks[time] + initial[0] * (1 - masks[time])
            c_next = c_next * masks[time] + initial[1] * (1 - masks[time])
            output.append(h_next)
            if drop_masks is not None: h_next = h_next * drop_masks
            hx = (h_next, c_next)
        output = torch.stack(output, 0)
        return output, hx

    @staticmethod
    def _forward_brnn(cell, input, masks, initial, drop_masks):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in reversed(range(max_time)):
            h_next, c_next = cell(input=input[time], hx=hx)
            h_next = h_next * masks[time] + initial[0] * (1 - masks[time])
            c_next = c_next * masks[time] + initial[1] * (1 - masks[time])
            output.append(h_next)
            if drop_masks is not None: h_next = h_next * drop_masks
            hx = (h_next, c_next)
        output.reverse()
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input, masks, initial=None):
        if self.batch_first:
            input = input.transpose(0, 1)
            masks = torch.unsqueeze(masks.transpose(0, 1), dim=2)
        max_time, batch_size, _ = input.size()
        masks = masks.expand(-1, -1, self.hidden_size)

        if initial is None:
            initial = input.data.new(batch_size, self.hidden_size).zero_()
            initial = (initial, initial)
        h_n = []
        c_n = []

        allhiddens = []
        for layer in range(self.num_layers):
            max_time, batch_size, input_size = input.size()
            input_mask, hidden_mask = None, None
            if self.training:
                input_mask = input.data.new(batch_size, input_size).fill_(1 - self.dropout_in)
                input_mask = torch.bernoulli(input_mask)
                input_mask = input_mask / (1 - self.dropout_in)
                input_mask = torch.unsqueeze(input_mask, dim=2).expand(-1, -1, max_time).permute(2, 0, 1)
                input = input * input_mask

                hidden_mask = input.data.new(batch_size, self.hidden_size).fill_(1 - self.dropout_out)
                hidden_mask = torch.bernoulli(hidden_mask)
                hidden_mask = hidden_mask / (1 - self.dropout_out)

            layer_output, (layer_h_n, layer_c_n) = LSTM._forward_rnn(cell=self.fcells[layer], \
                                                                     input=input, masks=masks, initial=initial,
                                                                     drop_masks=hidden_mask)
            if self.bidirectional:
                blayer_output, (blayer_h_n, blayer_c_n) = LSTM._forward_brnn(cell=self.bcells[layer], \
                                                                             input=input, masks=masks, initial=initial,
                                                                             drop_masks=hidden_mask)

            h_n.append(torch.cat([layer_h_n, blayer_h_n], 1) if self.bidirectional else layer_h_n)
            c_n.append(torch.cat([layer_c_n, blayer_c_n], 1) if self.bidirectional else layer_c_n)
            input = torch.cat([layer_output, blayer_output], 2) if self.bidirectional else layer_output
            allhiddens.append(input)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        return input, (h_n, c_n, allhiddens)


def last_pooling(inputs, masks):
    # inputs  sen_num x sent_len x hidden float32
    # masks   sen_num x sent_len          float32

    index = torch.sum(masks, 1).long() - 1  # last step

    batch_res = []
    for i in range(inputs.shape[0]):
        batch_res.append(torch.index_select(inputs[i], 0, index[i]))  # hidden*2

    reps = torch.cat(batch_res, 0)  # sen_num x hidden*2

    return reps
