# This file contains code derived from:
# - https://github.com/nathanael-fijalkow/DeepSynth (MIT License)
# Original authors: Nathanaël Fijalkow
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence

# Shape conventions:
# IO  = [[I1, ..., Ik], O]   (lists of integer tokens)
# IOs = [IO1, ..., IOn]      (one task)
# tasks = [(IOs_i, program_i), ...]


class SimpleEmbedding(nn.Module):
    def __init__(self,
                 IOEncoder,
                 output_dimension,
                 size_hidden,
                 ):
        super(SimpleEmbedding, self).__init__()

        self.IOEncoder = IOEncoder
        self.lexicon_size = IOEncoder.lexicon_size
        self.output_dimension = output_dimension

        embedding = nn.Embedding(self.lexicon_size, size_hidden)
        self.embedding = embedding

        self.hidden = nn.Sequential(
            nn.Linear(size_hidden, size_hidden),
            nn.LeakyReLU(),
            nn.Linear(size_hidden, output_dimension),
            nn.LeakyReLU(),
        )

    def forward_IOs(self, IOs):
        e = self.IOEncoder.encode_IOs(IOs)
        logging.debug("encoding size: {}".format(e.size()))
        e = self.embedding(e)
        logging.debug("embedding size: {}".format(e.size()))
        e = self.hidden(e)
        e = torch.mean(e, 0)
        assert(e.size() == (self.IOEncoder.output_dimension, self.output_dimension))
        return torch.flatten(e)

    def forward(self, batch_IOs):
        res = torch.stack([self.forward_IOs(IOs) for IOs in batch_IOs])
        assert(res.size() == (len(batch_IOs), self.IOEncoder.output_dimension * self.output_dimension))
        return res

class RNNEmbedding(nn.Module):
    def __init__(self,
                 IOEncoder,
                 output_dimension,
                 size_hidden,
                 number_layers_RNN,
                 ):
        super(RNNEmbedding, self).__init__()

        self.IOEncoder = IOEncoder
        self.lexicon_size = IOEncoder.lexicon_size
        self.output_dimension = output_dimension
        self.size_hidden = size_hidden

        embedding = nn.Embedding(self.lexicon_size, size_hidden)
        self.embedding = embedding

        Hin = size_hidden * IOEncoder.output_dimension
        Hout = IOEncoder.output_dimension * output_dimension
        self.RNN_layer = nn.GRU(Hin, Hout, number_layers_RNN, batch_first=True)

    def _forward_IOs(self, IOs):
        e = self.IOEncoder.encode_IOs(IOs)
        logging.debug("encoding size: {}".format(e.size()))
        e = self.embedding(e)
        logging.debug("embedding size: {}".format(e.size()))
        assert e.size() == (len(IOs), self.IOEncoder.output_dimension, self.size_hidden),\
         "size not equal to: {} {} {}".format(len(IOs), self.IOEncoder.output_dimension, self.size_hidden)
        e = torch.flatten(e, start_dim=1)
        e = torch.unsqueeze(e, 0)
        e, _ = self.RNN_layer(e)
        e = torch.squeeze(torch.squeeze(e, 0)[-1, :], 0)
        assert e.size() == (self.IOEncoder.output_dimension * self.output_dimension,),\
         "size not equal to: {}".format(self.IOEncoder.output_dimension * self.output_dimension)
        return e

    def forward(self, batch_IOs):
        res = torch.stack([self._forward_IOs(IOs) for IOs in batch_IOs])
        assert(res.size() == (len(batch_IOs), self.IOEncoder.output_dimension * self.output_dimension))
        return res
    
class RNNMatrixEmbedding(nn.Module):
    def __init__(self, IOEncoder, output_dimension, size_hidden, number_layers_RNN):
        super(RNNMatrixEmbedding, self).__init__()

        self.IOEncoder = IOEncoder
        self.size_hidden = size_hidden
        self.output_dimension = output_dimension

        self.embedding = nn.Embedding(self.IOEncoder.lexicon_size, size_hidden)

        Hin = size_hidden
        Hout = self.output_dimension
        self.RNN_layer = nn.GRU(Hin, Hout, number_layers_RNN, batch_first=True)

    def _forward_IOs(self, IOs):
        e = self.IOEncoder.encode_IOs(IOs)
        e = self.embedding(e)
        e = e.view(e.size(0), e.size(1), -1)
        e, _ = self.RNN_layer(e)
        e = e[:, -1, :]
        return e

    def forward(self, batch_IOs):
        return torch.stack([self._forward_IOs(IOs) for IOs in batch_IOs])