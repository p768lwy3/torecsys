from typing import List, Tuple, TypeVar, Union

class IndexField(object):

    def __init__(self, unk_index: Tuple[int, int] = (0, 0)):
        
        self._unk_index = unk_index[0]
        self.vocab_dict = dict([unk_index])
        self.inverse_vocab_dict= dict()
    
    def __len__(self) -> int:
        return len(self.vocab_dict)
    
    @property
    def unk_index(self) -> int:
        return self._unk_index
    
    @property
    def unk_token(self) -> int:
        return self.vocab_dict.get(self.unk_index)
    
    @property
    def current_max_token(self) -> int:
        return max(list(self.vocab_dict.values()))
    
    @unk_index.setter
    def unk_index(self, unk_index: int):
        if not isinstance(unk_index, int):
            raise TypeError(f"{type(unk_index).__name__}")
        
        # Pop original token of the index from vocab_dict
        unk_token = self.vocab_dict.pop(self.unk_index)
        
        # Set unk_index and bind unk_token to the unk_index
        self._unk_index = unk_index
        self.vocab_dict[self.unk_index] = unk_token
    
    @unk_token.setter
    def unk_token(self, unk_token: int):
        if not isinstance(unk_token, int):
            raise TypeError(f"{type(unk_token).__name__} not allowed.")
        
        # Set unk_token to vocab_dict
        self.vocab_dict[self.unk_index] = unk_token
    
    def build_vocab(self, dataset: List[List[int]]) -> TypeVar("IndexField"):

        # Get current max token in vocab_dict
        max_tkn = self.current_max_token

        if isinstance(dataset, list):
            # Loop through build_vocab
            for element in dataset:
                if isinstance(element, list) or isinstance(element, int):
                    self.build_vocab(element)
                else:
                    raise TypeError(f"{type(element).__name__} not allowed.")
        
        elif isinstance(dataset, int):
            max_tkn += 1
            if dataset not in self.vocab_dict:
                self.vocab_dict[dataset] = max_tkn
        
        else:
            raise TypeError(f"{type(dataset).__name__} not allowed.")
        
        # Update inverse_vocab_dict
        self._build_inverse_dict()
        
        return self
    
    def _build_inverse_dict(self):

        self.inverse_vocab_dict = {v: k for k, v in self.vocab_dict.items()}
    
    def from_idx_to_tkn(self, tokens: Union[int, List[int]]) -> Union[int, List[int]]:
        
        if isinstance(tokens, list):
            return [self.from_idx_to_tkn(tkn) for tkn in tokens]
        elif isinstance(tokens, int):
            indices = self.vocab_dict.get(tokens, self.vocab_dict[self.unk_index])
            return indices
        else:
            raise TypeError(f"{type(tokens).__name__} not allowed.")
    
    def from_tkn_to_idx(self, indices: Union[int, List[int]]) -> Union[int, List[int]]:

        if isinstance(indices, list):
            return [self.from_tkn_to_idx(idx) for idx in indices]
        elif isinstance(indices, int):
            tokens = self.inverse_vocab_dict.get(indices, self.unk_index)
            return tokens
        else:
            raise TypeError(f"{type(indices).__name__} not allowed.")
    
    def fit_predict(self, tokens: Union[int, List[int]]):

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

                # Update inverse_vocab_dict
                self._build_inverse_dict()

                return indices
        else:
            raise TypeError(f"{type(tokens).__name__} not allowed.")
        