# This file contains code derived from:
# - https://github.com/nathanael-fijalkow/DeepSynth (MIT License)
# Original authors: Nathanaël Fijalkow
from grammar.pcfg import PCFG
import random
import logging 
import torch
from type_system import STRING, Type, List, INT, BOOL
from cons_list import tuple2constlist

from flashfill_dataset_loader import randomWord

class Dataset(torch.utils.data.IterableDataset):
    """Iterable dataset producing a stream of (IOs, program) tasks."""

    def __init__(self,
        size,
        dsl,
        pcfg_dict,
        nb_examples_max,
        arguments,
        ProgramEncoder,
        size_max,
        lexicon,
        for_flashfill=False
        ):
        super(Dataset).__init__()
        self.size = size
        self.dsl = dsl
        self.input_sampler = Input_sampler(size_max=size_max, lexicon=lexicon)
        self.program_sampler = {t: pcfg.sampling() for t, pcfg in pcfg_dict.items()}
        self.arguments = arguments
        self.allowed_types = list(self.program_sampler.keys())

        self.nb_examples_max = nb_examples_max
        self.ProgramEncoder = ProgramEncoder
        self.lexicon = lexicon
        self.for_flashfill = for_flashfill

    def __iter__(self):
        return (self.__single_data__() for i in range(self.size))

    def __single_data__(self):
        flag = True
        output = None
        while flag or output == None:
            selected_type: Type = random.choice(self.allowed_types)
            rtype = selected_type.returns()
            program = next(self.program_sampler[selected_type])
            while program.is_constant():
                program = next(self.program_sampler[selected_type])

            if self.for_flashfill:
                n = program.count_constants()
                constants = [randomWord() for _ in range(n)]
                program = program.derive_with_constants(constants)

            nb_IOs = random.randint(1, self.nb_examples_max)
            inputs = [[self.input_sampler.sample(type_) for type_ in self.arguments[selected_type]] for _ in range(nb_IOs)]

            try:
                outputs = []
                for input_ in inputs:
                    environment = tuple2constlist(input_)
                    output = program.eval_naive(self.dsl, environment)
                    if self.__output_validation__(output, rtype):
                        outputs.append(output)
                    else:
                        raise ValueError()
                flag = False
            except:
                pass
        IOs = [[I, O] for I, O in zip(inputs, outputs)]
        logging.debug('Found a program:\n{}\nand inputs:\n{}'.format(program, IOs))
        return IOs, self.ProgramEncoder(program), program, selected_type

    def __output_validation__(self, output, return_type: Type):
        if isinstance(output, int) and return_type == INT:
            return True
        if len(output) == 0 or len(output) > self.input_sampler.size_max:
            return False
        for e in output:
            if e not in self.lexicon:
                return False
        return True


class Input_sampler():
    """Sample values for INT, STRING, BOOL, and List-of-those types from a single shared lexicon."""

    def __init__(self, size_max, lexicon) -> None:
        self.size_max = size_max
        self.lexicon = lexicon

    def sample(self, type: Type):
        if isinstance(type, List):
            size = random.randint(1, self.size_max)
            res = [self.sample(type.type_elt) for _ in range(size)]
            return res
        if type.__eq__(INT):
            return random.choice(self.lexicon)
        elif type.__eq__(STRING):
            size = random.randint(1, self.size_max)
            res = "".join([random.choice(self.lexicon) for _ in range(size)])
            return res
        elif type.__eq__(BOOL):
            return random.random() > .5
        assert(False)

