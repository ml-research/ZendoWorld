import os
import typing
from experiment_helper import __get_type_request
import torch
from type_system import BOOL, INT, STRING, Arrow, List, Type
from typing import Dict, Set, Tuple
from grammar.cfg import CFG
import grammar.dsl as dsl
from DSL import zendo
from Predictions.IOencodings import FixedSizeEncoding, ZendoFixedSizeEncoding
from Predictions.embeddings import RNNEmbedding, RNNMatrixEmbedding, SimpleEmbedding
from Predictions.models import RulesPredictor, BigramsPredictor, NNDictRulesPredictor


def __block__(input_dim, output_dimension, activation):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dimension),
        activation,
    )

def get_model_name(model) -> str:
    name: str = ""
    if isinstance(model.IOEncoder, FixedSizeEncoding):
        name += "fixed"
    else:
        name += "variable"
    if isinstance(model.IOEmbedder, SimpleEmbedding):
        name += "+simple"
    else:
        name += "+rnn"
    if isinstance(model, NNDictRulesPredictor):
        name += "+nndict_rules"
    elif isinstance(model, RulesPredictor):
        name += "+rules"
    else:
        name += "+bigrams"
    return name


def __buildintlist_model(dsl: dsl.DSL, max_program_depth: int, nb_arguments_max: int, lexicon: typing.List[int], size_max: int, size_hidden: int, embedding_output_dimension: int, number_layers_RNN: int) -> Tuple[CFG, RulesPredictor]:
    type_request = Arrow(List(INT), List(INT))
    cfg = dsl.DSL_to_CFG(
        type_request, max_program_depth=max_program_depth)

    IOEncoder = FixedSizeEncoding(
        nb_arguments_max=nb_arguments_max,
        lexicon=lexicon,
        size_max=size_max,
    )

    IOEmbedder = RNNEmbedding(
        IOEncoder=IOEncoder,
        output_dimension=embedding_output_dimension,
        size_hidden=size_hidden,
        number_layers_RNN=number_layers_RNN,
    )

    latent_encoder = torch.nn.Sequential(
        __block__(IOEncoder.output_dimension * IOEmbedder.output_dimension, size_hidden, torch.nn.Sigmoid()),
        __block__(size_hidden, size_hidden, torch.nn.Sigmoid()),
    )

    model = RulesPredictor(
        cfg=cfg,
        IOEncoder=IOEncoder,
        IOEmbedder=IOEmbedder,
        latent_encoder=latent_encoder,
    )

    return cfg, model

def __buildintlist_zendo_model(dsl: dsl.DSL, max_program_depth: int, size_max: int, size_hidden: int, embedding_output_dimension: int, number_layers_RNN: int, autoload=False, name="variable+rnn+rules_zendo.weights") -> Tuple[CFG, RulesPredictor]:
    type_request = Arrow(List(zendo.PIECE), BOOL)
    cfg = dsl.DSL_to_CFG(
        type_request, max_program_depth=max_program_depth)

    IOEncoder = ZendoFixedSizeEncoding(
        size_max=11,
        lexicon=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    IOEmbedder = RNNMatrixEmbedding(
        IOEncoder=IOEncoder,
        output_dimension=embedding_output_dimension,
        size_hidden=size_hidden,
        number_layers_RNN=number_layers_RNN,
    )

    latent_encoder = torch.nn.Sequential(
        __block__(IOEncoder.output_dimension, size_hidden, torch.nn.Sigmoid()),
        __block__(size_hidden, size_hidden, torch.nn.Sigmoid()),
    )

    model = RulesPredictor(
        cfg=cfg,
        IOEncoder=IOEncoder,
        IOEmbedder=IOEmbedder,
        latent_encoder=latent_encoder,
    )

    if autoload:
        weights_file = name
        if os.path.exists(weights_file):
            model.load_state_dict(torch.load(weights_file, weights_only=True))
            print("Loaded weights.")

    return cfg, model


def build_deepcoder_intlist_model(max_program_depth: int = 4, autoload: bool = True) -> Tuple[dsl.DSL, CFG, RulesPredictor]:
    size_max = 10
    nb_arguments_max = 1
    lexicon = [x for x in range(-256, 256)]

    embedding_output_dimension = 10
    number_layers_RNN = 1
    size_hidden = 64
    deepcoder_dsl = dsl.DSL(deepcoder.semantics, deepcoder.primitive_types, deepcoder.no_repetitions)

    deepcoder_cfg, model = __buildintlist_model(
        deepcoder_dsl, max_program_depth, nb_arguments_max, lexicon, size_max, size_hidden, embedding_output_dimension, number_layers_RNN)

    if autoload:
        weights_file = get_model_name(model) + "_deepcoder.weights"
        if os.path.exists(weights_file):
            model.load_state_dict(torch.load(weights_file), weights_only=True)
            print("Loaded weights.")

    return deepcoder_dsl, deepcoder_cfg, model


def __build_generic_model(dsl: dsl.DSL, cfg_dictionary: Dict[Type, CFG], nb_arguments_max: int, lexicon: typing.List[int], size_max: int, size_hidden: int, embedding_output_dimension: int, number_layers_RNN: int) -> BigramsPredictor:
    IOEncoder = FixedSizeEncoding(
        nb_arguments_max=nb_arguments_max,
        lexicon=lexicon,
        size_max=size_max,
    )
    IOEmbedder = RNNEmbedding(
        IOEncoder=IOEncoder,
        output_dimension=embedding_output_dimension,
        size_hidden=size_hidden,
        number_layers_RNN=number_layers_RNN,
    )

    latent_encoder = torch.nn.Sequential(
        __block__(IOEncoder.output_dimension *
                  IOEmbedder.output_dimension, IOEncoder.output_dimension *
                  IOEmbedder.output_dimension // 2, torch.nn.ReLU()),
    )

    return BigramsPredictor(
        cfg_dictionary=cfg_dictionary,
        primitive_types={x: x.type for x in dsl.list_primitives},
        IOEncoder=IOEncoder,
        IOEmbedder=IOEmbedder,
        latent_encoder=latent_encoder
    )

def __build_generic_zendo_model(dsl: dsl.DSL, max_program_depth: int, size_max: int, size_hidden: int, embedding_output_dimension: int, number_layers_RNN: int, autoload=False, name="variable+simple+bigrams_zendo.weights") -> Tuple[CFG, RulesPredictor]:
    nb_arguments_max = 2
    type_request = Arrow(List(zendo.PIECE), BOOL)
    dsl.instantiate_polymorphic_types()
    requests = dsl.all_type_requests(nb_arguments_max)
    cfg_dict = {}
    for type_req in requests:
        # Skip nested lists.
        if any(ground_type.size() >= 3 for ground_type in type_req.list_ground_types()):
            print(f"Skipping type {type_req} because it contains a list of lists.")
            continue
        try:
            cfg_dict[type_req] = dsl.DSL_to_CFG(
                type_req, max_program_depth=max_program_depth)
        except Exception as e:
            continue
    cfg = dsl.DSL_to_CFG(
        type_request, max_program_depth=max_program_depth)

    IOEncoder = ZendoFixedSizeEncoding(
        size_max=11,
        lexicon=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    IOEmbedder = SimpleEmbedding(
        IOEncoder=IOEncoder,
        output_dimension=64,
        size_hidden=64,
    )
    latent_encoder = torch.nn.Sequential(
        __block__(64 * IOEncoder.output_dimension, 128, torch.nn.ReLU()),
        __block__(128, 64, torch.nn.ReLU()),
    )

    model = BigramsPredictor(
        cfg_dictionary=cfg_dict,
        primitive_types={x: x.type for x in dsl.list_primitives},
        IOEncoder=IOEncoder,
        IOEmbedder=IOEmbedder,
        latent_encoder=latent_encoder,
    )

    if autoload:
        weights_file = name
        if os.path.exists(weights_file):
            print(f"Loading weights from {weights_file}")
            model.load_state_dict(torch.load(weights_file, weights_only=True))
            print("Loaded weights.")
        else:
            print(f"Warning: Weights file {weights_file} not found. Skipping loading weights.")
            raise FileNotFoundError(f"Weights file {weights_file} not found.")

    return cfg, model