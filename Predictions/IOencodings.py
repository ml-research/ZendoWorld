# This file contains code derived from:
# - https://github.com/nathanael-fijalkow/DeepSynth (MIT License)
# Original authors: Nathanaël Fijalkow
import torch


class FixedSizeEncoding():
    """Fixed-size IO encoding for lists.

    Each input/output is encoded as ``2 * size_max`` longs: alternating
    (value_idx, NOTPAD/PAD-flag). Up to ``nb_arguments_max`` inputs are concatenated,
    followed by the output. Empty positions are filled with the PAD index.

    Example with ``size_max=2``, ``nb_arguments_max=3``, ``IO=[[[11,20],[3]], [12,2]]``::

        [11,1,20,1, 3,1,0,0, 0,0,0,0, 12,1,2,1]
    """

    def __init__(self,
                 nb_arguments_max,
                 lexicon,
                 size_max,
                 ) -> None:
        self.nb_arguments_max = nb_arguments_max
        self.size_max = size_max
        self.output_dimension = 2 * size_max * (1 + nb_arguments_max)
        self.lexicon = lexicon[:]
        self.lexicon += ["PAD", "NOTPAD"]
        self.lexicon_size = len(self.lexicon)
        self.symbolToIndex = {
            symbol: index for index, symbol in enumerate(self.lexicon)
        }

    def _encode_single_arg(self, arg):
        if isinstance(arg, int):
            arg = [arg]
        res = torch.zeros(2*self.size_max, dtype=torch.long)
        res += self.symbolToIndex["PAD"]
        if len(arg) > self.size_max:
            assert False, \
                "IOEncodings.py: FixedSizeEncoding: This input is too long: len({})={} > {}".format(arg, len(arg), self.size_max)
        for i, e in enumerate(arg):
            res[2*i] = self.symbolToIndex[e]
            res[2*i+1] = self.symbolToIndex["PAD"]
        return res

    def encode_IO(self, IO):
        res = []
        inputs, output = IO
        if len(inputs) > self.nb_arguments_max:
            assert False, \
                "IOEncodings.py: FixedSizeEncoding: Too many inputs: len({})={} > {}".format(
                    inputs, len(inputs), self.nb_arguments_max)
        for i in range(self.nb_arguments_max):
            try:
                input_ = inputs[i]
                embedded_input = self._encode_single_arg(input_)
                res.append(embedded_input)
            except:
                not_pad_tensor = torch.zeros(2*self.size_max, dtype=torch.long)
                not_pad_tensor += self.symbolToIndex["PAD"]
                res.append(not_pad_tensor)
        res.append(self._encode_single_arg(output))
        res = torch.cat(res)
        return res

    def encode_IOs(self, IOs):
        res = []
        for IO in IOs:
            res.append(self.encode_IO(IO))
        res = torch.stack(res)
        return res


class ZendoFixedSizeEncoding():
    """Fixed-size encoding for Zendo: 7x11 categorical matrix + 1 boolean label."""

    def __init__(self,
                 size_max,
                 lexicon,
                 ) -> None:
        self.size_max = size_max
        self.output_dimension = size_max * 7 + 1
        self.symbolToIndex = {i: i for i in range(9)}
        self.lexicon_size = len(self.symbolToIndex)
        self.max_value = 640

    def _encode_single_arg(self, arg):
        res = torch.zeros(self.size_max, dtype=torch.long)
        for i, e in enumerate(arg[:11]):
            res[i] = e
        return res

    def encode_IO(self, IO):
        res = []
        inputs, output = IO
        if len(inputs) > 7:
            print("Warning: Too many inputs, truncating to 7.", inputs, output)
        for input_ in inputs:
            encoded_input = self._encode_single_arg(input_)
            res.append(encoded_input)

        output_tensor = torch.tensor([output], dtype=torch.long)
        res.append(output_tensor)
        res = torch.cat(res)
        return res

    def encode_IOs(self, IOs):
        res = []
        for IO in IOs:
            encoded = self.encode_IO(IO)
            res.append(encoded)
        res = torch.stack(res)
        return res