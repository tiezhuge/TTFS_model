
import logging

import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
class Dictionary:
    """A mapping from symbols to consecutive integers"""

    def __init__(
            self,
            *,  # begin keyword-only arguments
            bos="[CLS]",
            pad="[PAD]",
            eos="[SEP]",
            unk="[UNK]",
            extra_special_symbols=None,
    ):
        # print("dictionary中的 def __init__(:")
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.specials = set()
        self.specials.add(bos)
        self.specials.add(unk)
        self.specials.add(pad)
        self.specials.add(eos)

    def __eq__(self, other):
        # print("dictionary中的def __eq__(self, other):")
        return self.indices == other.indices

    def __getitem__(self, idx):
        # print("dictionary中的def __getitem__(self, idx):")
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        # print("dictionary中的 def __len__(self)::")
        """Returns the number of symbols in the dictionary"""
        print("dictionary中的 self.symbols： ", self.symbols)
        return len(self.symbols)

    def __contains__(self, sym):
        # print("dictionary中的 def __contains__(self, sym)::")
        return sym in self.indices

    def vec_index(self, a):
        # print("dictionary中的 def vec_index(self, a):")
        return np.vectorize(self.index)(a)

    def index(self, sym):
        # print("dictionary中的 def index(self, sym):")
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.indices[self.unk_word]

    def special_index(self):
        # print("dictionary中的 def special_index(self):")
        return [self.index(x) for x in self.specials]

    def add_symbol(self, word, n=1, overwrite=False, is_special=False):
        # print("dictionary中的 def add_symbol:")
        """Adds a word to the dictionary"""
        if is_special:
            self.specials.add(word)
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        # print("dictionary中的 def bos(self):")
        return self.index(self.bos_word)

    def pad(self):
        """Helper to get index of pad symbol"""
        # print("dictionary中的 def pad(self):")
        return self.index(self.pad_word)

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        # print("dictionary中的 def  def eos(self):")
        return self.index(self.eos_word)

    def unk(self):
        """Helper to get index of unk symbol"""
        # print("dictionary中的 def def unk(self)::")
        return self.index(self.unk_word)

    @classmethod
    def load(cls, f):
        print("dictionary中的 def load(cls, f):")
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f)

        print("d.add_from_file(f) f: ", f)  # /home/ubuntu/my_code/feng/molecular_property_prediction/dict.txt
        # print("d.add_from_file(f): ",d.add_from_file(f))
        print("load_return  cls: ", cls)  # <class 'unicore.data.dictionary.Dictionary'>
        print("load_return  cls(): ", cls())  # <unicore.data.dictionary.Dictionary object at 0x7f237d99bd60>
        print("load_return  d: ", d)  # <unicore.data.dictionary.Dictionary object at 0x7f81bdd6cd00>
        print("len(d): ", len(d))
        return d

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        # print("dictionary中的 def add_from_file(self, f):")
        if isinstance(f, str):
            try:
                with open(f, "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )
            return

        lines = f.readlines()

        for line_idx, line in enumerate(lines):
            try:
                splits = line.rstrip().rsplit(" ", 1)
                line = splits[0]
                field = splits[1] if len(splits) > 1 else str(len(lines) - line_idx)
                if field == "#overwrite":
                    overwrite = True
                    line, field = line.rsplit(" ", 1)
                else:
                    overwrite = False
                count = int(field)
                word = line
                if word in self and not overwrite:
                    logger.info(
                        "Duplicate word found when loading Dictionary: '{}', index is {}.".format(word,
                                                                                                  self.indices[word])
                    )
                else:
                    self.add_symbol(word, n=count, overwrite=overwrite)
            except ValueError:
                raise ValueError(
                    "Incorrect dictionary format, expected '<token> <cnt> [flags]'"
                )
