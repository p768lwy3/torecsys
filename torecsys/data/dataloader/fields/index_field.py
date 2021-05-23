"""

"""

from typing import TypeVar

from torecsys.data.dataloader.fields import Field
from torecsys.utils.typing_extensions import Ints


class IndexField(Field):
    """
    IndexField is a field for dataloader.collate_fn to convert between index and token
    """
    IndexField = TypeVar('IndexField')

    def __init__(self, unk_index: int = 0, unk_token: int = 0):
        """
        Initializer of index field

        Args:
            unk_index: index of unknown
            unk_token: token of unknown
        """
        self._unk_index = unk_index
        self.vocab_dict = {self._unk_index: unk_token}
        self.inverse_vocab_dict = {}

    def __len__(self) -> int:
        """
        Get the size of vocabulary dictionary

        Returns:
            int: Total number of token in the vocabulary dictionary
        """
        return len(self.vocab_dict)

    @property
    def unk_index(self) -> int:
        """
        Property of unk_index

        Returns:
            int: Value of unk_index
        """
        return self._unk_index

    @unk_index.setter
    def unk_index(self, unk_index: int):
        """
        Setter of unk_index

        Args:
            unk_index (int): index of unknown token
        """
        if not isinstance(unk_index, int):
            raise TypeError(f'Type of unk_index {type(unk_index).__name__} is not allowed')

        unk_token = self.vocab_dict.pop(self.unk_index)
        self._unk_index = unk_index
        self.vocab_dict[self.unk_index] = unk_token

    @property
    def unk_token(self) -> int:
        """
        Property of unk_token

        Returns:
            int. Value of unk_token
        """
        return self.vocab_dict.get(self.unk_index)

    @unk_token.setter
    def unk_token(self, unk_token: int):
        if not isinstance(unk_token, int):
            raise TypeError(f'Type of unk_token {type(unk_token).__name__} is not allowed')

        self.vocab_dict[self.unk_index] = unk_token

    @property
    def current_max_token(self) -> int:
        """
        Property of current_max_token

        Returns:
            int. Value of current_max_token
        """
        return max(list(self.vocab_dict.values()))

    def _build_inverse_dict(self):
        self.inverse_vocab_dict = {v: k for k, v in self.vocab_dict.items()}

    def build_vocab(self, dataset: Ints) -> IndexField:
        """
        Build vocabulary with input dataset

        Args:
            dataset:

        Returns:

        """
        max_tkn = self.current_max_token

        if isinstance(dataset, list):
            for element in dataset:
                if isinstance(element, list) or isinstance(element, int):
                    self.build_vocab(element)
                else:
                    raise TypeError(f'Type of element {type(element).__name__} is not allowed')

        elif isinstance(dataset, int):
            max_tkn += 1
            if dataset not in self.vocab_dict:
                self.vocab_dict[dataset] = max_tkn

        else:
            raise TypeError(f'Type of dataset {type(dataset).__name__} is not allowed')

        self._build_inverse_dict()
        return self

    def indices(self, tokens: Ints) -> Ints:
        if isinstance(tokens, list):
            return [self.indices(tkn) for tkn in tokens]
        elif isinstance(tokens, int):
            indices = self.vocab_dict.get(tokens, self.vocab_dict[self.unk_index])
            return indices
        else:
            raise TypeError(f'Type of tokens {type(tokens).__name__} is not allowed')

    def tokens(self, indices: Ints) -> Ints:
        if isinstance(indices, list):
            return [self.tokens(idx) for idx in indices]
        elif isinstance(indices, int):
            tokens = self.inverse_vocab_dict.get(indices, self.unk_index)
            return tokens
        else:
            raise TypeError(f'Type of indices {type(indices).__name__} is not allowed')

    def fit_predict(self, tokens: Ints) -> Ints:
        if isinstance(tokens, list):
            return [self.fit_predict(tkn) for tkn in tokens]
        elif isinstance(tokens, float):
            return self.fit_predict(int(tokens))
        elif isinstance(tokens, int):
            if tokens not in self.vocab_dict:
                max_tkn = self.current_max_token + 1
                self.vocab_dict[tokens] = max_tkn
                return max_tkn
            else:
                indices = self.vocab_dict[tokens]
                self._build_inverse_dict()
                return indices

        else:
            raise TypeError(f'Type of tokens {type(tokens).__name__} is not allowed')
