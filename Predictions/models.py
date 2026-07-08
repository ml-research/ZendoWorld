# This file contains code derived from:
# - https://github.com/nathanael-fijalkow/DeepSynth (MIT License)
# Original authors: Nathanaël Fijalkow
import torch
from torch import nn
import numpy as np

import copy

from grammar.pcfg import PCFG
from grammar.pcfg_logprob import LogProbPCFG
from program import Function, Variable, BasicPrimitive, New
from type_system import BOOL

device = 'cpu'


def guess_output_size(latent_encoder, H):
    test_input = torch.zeros(1, H)
    with torch.no_grad():
        output = latent_encoder(test_input)
        output_size = output.size(1)
    return output_size


class RulesPredictor(nn.Module):
    """Predicts a flat probability per (S, P) production from a CFG template."""

    def __init__(self,
                 cfg,
                 IOEncoder,
                 IOEmbedder,
                 latent_encoder,
                 ):
        super(RulesPredictor, self).__init__()

        self.cfg = cfg
        self.IOEncoder = IOEncoder
        self.IOEmbedder = IOEmbedder
        self.latent_encoder = latent_encoder

        H = IOEmbedder.output_dimension
        output_size = guess_output_size(latent_encoder, H)

        self.loss = torch.nn.BCELoss(reduction='mean')

        self.init_RuleToIndex()

        self.final_layer = nn.Sequential(
            nn.Linear(output_size, self.output_dimension),
            nn.Sigmoid(),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def metrics(self, **kwargs):
        return {}

    def init_RuleToIndex(self):
        self.output_dimension = 0

        index = 0
        # RuleToIndex[(S, P)]: position of derivation (S, P) in the final layer.
        self.RuleToIndex = {}
        for S in self.cfg.rules:
            self.output_dimension += len(self.cfg.rules[S])
            for P in self.cfg.rules[S]:
                self.RuleToIndex[(S, P)] = index
                index += 1

    def forward(self, batch_IOs):
        x = self.IOEmbedder.forward(batch_IOs)
        x = self.latent_encoder(x)
        x = self.final_layer(x)
        return x

    def reconstruct_grammars(self, batch_predictions):
        res = []
        for x in batch_predictions:
            rules = {}
            for S in self.cfg.rules:
                rules[S] = {}
                for P in self.cfg.rules[S]:
                    cpy_P = copy.deepcopy(P)
                    rules[S][cpy_P] = self.cfg.rules[S][P], \
                        float(x[0][self.RuleToIndex[(S, P)]])
            grammar = PCFG(
                start=self.cfg.start,
                rules=rules,
                max_program_depth=self.cfg.max_program_depth,
                clean=True)
            res.append(grammar)
        return res

    def ProgramEncoder(self, program, S=None, tensor=None):
        """Indicator over CFG transitions: 1 for those used to derive ``program``, else 0."""
        if S == None:
            S = self.cfg.start

        if tensor == None:
            tensor = torch.zeros(self.output_dimension)

        if isinstance(program, Function):
            F = program.function
            args_P = program.arguments
            tensor[self.RuleToIndex[(S, F)]] = 1
            for i, arg in enumerate(args_P):
                self.ProgramEncoder(arg, self.cfg.rules[S][F][i], tensor)

        if isinstance(program, (BasicPrimitive, Variable)):
            tensor[self.RuleToIndex[(S, program)]] = 1

        assert(tensor.size() == (self.output_dimension,))
        return tensor

    def custom_collate(self, batch):
        print("custom collate for RulesPredictor")
        return [batch[i][0] for i in range(len(batch))], torch.stack([batch[i][1] for i in range(len(batch))])


class NNDictRulesPredictor(nn.Module):
    """RulesPredictor variant that uses one log-softmax projection head per non-terminal."""

    def __init__(self,
                 cfg,
                 IOEncoder,
                 IOEmbedder,
                 latent_encoder
                 ):
        super(NNDictRulesPredictor, self).__init__()

        self.cfg = cfg
        self.IOEncoder = IOEncoder
        self.IOEmbedder = IOEmbedder
        self.latent_encoder = latent_encoder

        self.loss = lambda batch_grammar, batch_program:\
            - sum(grammar.log_probability_program(grammar.start, program)
                  for grammar, program in zip(batch_grammar, batch_program))

        H = IOEncoder.output_dimension * self.IOEmbedder.output_dimension
        output_size = guess_output_size(latent_encoder, H)

        projection_layer = {}
        for S in self.cfg.rules:
            n_productions = len(self.cfg.rules[S])
            module = nn.Sequential(nn.Linear(output_size, n_productions),
                                   nn.LogSoftmax(-1))
            projection_layer[str(S)] = module
        self.projection_layer = nn.ModuleDict(projection_layer)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.1)

    def metrics(self, loss: float, batch_size: int, **kwargs):
        return {"average probability": np.exp(- loss / batch_size)}

    def ProgramEncoder(self, program):
        return program

    def forward(self, batch_IOs):
        grammars = []
        for x in batch_IOs:
            x = self.IOEmbedder.forward([x])
            x = self.latent_encoder(x)
            probabilities = {S: self.projection_layer[format(S)](x)
                             for S in self.cfg.rules}
            rules = {}
            for S in self.cfg.rules:
                rules[S] = {}
                for j, P in enumerate(self.cfg.rules[S]):
                    cpy_P = copy.deepcopy(P)
                    rules[S][cpy_P] = self.cfg.rules[S][P], \
                        probabilities[S][0, j]
            grammar = LogProbPCFG(self.cfg.start,
                                  rules,
                                  max_program_depth=self.cfg.max_program_depth)
            grammar.clean()
            grammars.append(grammar)
        return grammars

    def custom_collate(self, batch):
        return [batch[i][0] for i in range(len(batch))], torch.stack([batch[i][1] for i in range(len(batch))])


class BigramsPredictor(nn.Module):
    """Predicts a (parent, arg_position, primitive) bigram log-probability tensor used to build per-type PCFGs."""

    def __init__(self,
                 cfg_dictionary,
                 primitive_types,
                 IOEncoder,
                 IOEmbedder,
                 latent_encoder,
                 variable_probability=0.2
                 ):
        super(BigramsPredictor, self).__init__()

        self.cfg_dictionary = cfg_dictionary
        self.primitive_types = primitive_types
        self.IOEncoder = IOEncoder
        self.IOEmbedder = IOEmbedder
        self.latent_encoder = latent_encoder

        self.variable_probability = variable_probability

        self.loss = lambda batch_grammar, batch_program:\
            - sum(grammar.log_probability_program(grammar.start, program)
                  for grammar, program in zip(batch_grammar, batch_program))

        H = IOEncoder.output_dimension * self.IOEmbedder.output_dimension
        output_size = guess_output_size(latent_encoder, H)
        print("Output size for BigramsPredictor:", output_size)

        self.symbolToIndex = {
            symbol: index for index, symbol in enumerate(self.primitive_types.keys())
        }

        # Variables are not predicted; their probability is assigned post-hoc.
        self.number_of_primitives = len(self.primitive_types)
        # +1 parent slot for the "no parent" case (the start symbol).
        self.number_of_parents = self.number_of_primitives + 1
        self.maximum_arguments = max(len(t.arguments())
                                     for t in self.primitive_types.values())
        self.q_predictor = nn.Linear(output_size,
                                     self.number_of_parents*self.maximum_arguments*self.number_of_primitives)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.1)

    def metrics(self, loss: float, batch_size: int, **kwargs) :
        return {"average probability": np.exp(- loss / batch_size)}


    def forward(self, batch_IOs):
        """Encode IOs into a (B, parents, args, primitives) log-softmax tensor used to build PCFGs."""
        x = self.IOEmbedder.forward(batch_IOs)
        x = self.latent_encoder(x)
        x = self.q_predictor.forward(x).view(-1,
                                             self.number_of_parents, self.maximum_arguments, self.number_of_primitives)
        x = nn.LogSoftmax(-1)(x)
        return x

    def reconstruct_grammars(self, batch_predictions, batch_type_requests, tensors=True):
        grammars = []
        for x, type_request in zip(batch_predictions, batch_type_requests):

            cfg = self.cfg_dictionary[type_request]

            rules = {}
            for S in cfg.rules:
                rules[S] = {}
                if S[1]:
                    parent_index = self.symbolToIndex[S[1][0]]
                    argument_number = S[1][1]
                else:
                    # Start symbol uses the "no parent" slot.
                    parent_index = self.number_of_primitives
                    argument_number = 0
                variables = []
                for P in cfg.rules[S]:
                    cpy_P = copy.deepcopy(P)
                    if isinstance(P, (BasicPrimitive, New)):
                        primitive_index = self.symbolToIndex[P]
                        if tensors:
                            rules[S][cpy_P] = cfg.rules[S][P], \
                                x[parent_index, argument_number, primitive_index]
                        else:
                            rules[S][cpy_P] = cfg.rules[S][P], \
                                x[parent_index, argument_number,
                                    primitive_index].item()
                    else:
                        # Variables get a placeholder; their probability is filled in below.
                        rules[S][cpy_P] = cfg.rules[S][P], -1
                        variables.append(cpy_P)
                # Variables hold a fixed share `variable_probability`, split uniformly among them;
                # the rest of the productions are renormalised to (1 - variable_probability).
                if variables:
                    if tensors:
                        total = sum(np.exp(rules[S][P][1].item())
                                    for P in rules[S] if P not in variables)
                    else:
                        total = sum(np.exp(rules[S][P][1])
                                    for P in rules[S] if P not in variables)

                    var_probability = self.variable_probability
                    if total > 0:
                        to_add = np.log(
                            1 - self.variable_probability) - np.log(total)
                        for P in rules[S]:
                            rules[S][P] = rules[S][P][0], rules[S][P][1] + to_add
                    else:
                        var_probability = 1
                    normalised_variable_logprob = np.log(
                        var_probability / len(variables))
                    for P in variables:
                        rules[S][P] = rules[S][P][0], normalised_variable_logprob
                        normalised_variable_logprob = np.log(
                            np.exp(normalised_variable_logprob) - 1e-7)
                else:
                    # Renormalise: not all DSL derivations are reachable from S.
                    epsilon = 1e-8
                    total = sum(np.exp(rules[S][P][1].item()) for P in rules[S])
                    total = max(total, epsilon)
                    to_add = np.log(1 / total)
                    for O in rules[S]:
                        rules[S][O] = rules[S][O][0], rules[S][O][1] + to_add
            grammar = LogProbPCFG(cfg.start,
                                  rules,
                                  max_program_depth=cfg.max_program_depth)
            grammar.clean()
            grammars.append(grammar)
        return grammars

    def ProgramEncoder(self, program):
        return program

    def custom_collate(self, batch):
        return [batch[i][0] for i in range(len(batch))], torch.stack([batch[i][1] for i in range(len(batch))])