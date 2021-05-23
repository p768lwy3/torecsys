"""

"""

from collections import Counter
from typing import Any, Callable, List, Tuple, TypeVar

import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils

from torecsys.data.dataloader.fields import Field


class SentenceField(Field):
    """
    SentenceField is a field for dataloader.collate_fn to parse string sentence
    """
    SentenceField = TypeVar('SentenceField')

    def __init__(self,
                 tokenize: Callable[[str], list] = lambda x: x.split(),
                 threshold: int = 0,
                 pad_token: Tuple[str, int] = ('<pad>', 0),
                 unk_token: Tuple[str, int] = ('<unk>', 1)):
        """
        Initializer of sentence field
        
        Args:
            tokenize (Callable[[str], list], optional): function to split input sentences to list of string.
                Defaults to lambda x:x.split().
            threshold (int, optional): threshold for dropping tokens. Defaults to 0.
            pad_token (Tuple[str, int], optional): padding token and corresponding index. Defaults to ('<pad>', 0).
            unk_token (Tuple[str, int], optional): unknown token and corresponding index. Defaults to ('<unk>', 1).
        """
        self.tokenize = tokenize
        self.threshold = threshold

        self.vocab_dict = dict([pad_token, unk_token])
        self.pad_token = pad_token[0]
        self.unk_token = unk_token[0]

        self.vocab_count = {}

        self.is_initialized = False

    def __len__(self) -> int:
        """
        Get the size of vocabulary dictionary
        
        Returns:
            int: Size of vocabulary dictionary
        """
        return len(self.vocab_dict)

    def _update_inverse_dict(self):
        """
        Build inverse dictionary where the key is index and the value is string token
        """
        self.inverse_vocab_dict = {v: k for k, v in self.vocab_dict.items()}

    def build_vocab(self, dataset: List[str], verbose: bool = False) -> SentenceField:
        """
        Build vocabulary with input dataset
        
        Args:
            dataset (List[str]): list of sentence strings for building the vocabulary
            verbose (bool): boolean flag to control whether the log is printed. Defaults to False.
        """
        max_id = max(list(self.vocab_dict.values()))
        counts = dict(Counter(tok for sent in dataset for tok in self.tokenize(sent)))

        appended = 0
        updated = 0
        for k, v in counts.items():
            if k not in self.vocab_dict and v > self.threshold:
                max_id += 1
                self.vocab_count[k] = v
                self.vocab_dict[k] = max_id
                appended += 1
            elif k in self.vocab_count:
                self.vocab_count[k] += v
                updated += 1
            else:
                continue

        if verbose:
            print(f'Total number of vocabulary added : {appended}.\nTotal number of vocabulary updated : {updated}.')

        self._update_inverse_dict()
        self.is_initialized = True
        return self

    def from_index(self,
                   inputs: torch.Tensor,
                   join_function: Callable[[List[str]], str] = ''.join) -> List[str]:
        """
        Return list of sentence string from a torch.Tensor of corresponding indices

        Args:
            inputs (torch.Tensor, shape = (batch size, max sequence length, ), data_type = torch.long): corresponding
                index of tokens in sentence strings
            join_function (Callable[[List[str]], str], optional): function to join the token string into a sentence.
                Defaults to ''.join.

        Raises:
            ValueError: when the embedder dimension is not equal to 2, i.e. (batch size, max sequence length, )

        Returns:
            List[str]: list of sentence strings
        """
        # check whether dimension of embedder is 2
        if inputs.dim() != 2:
            raise ValueError('embedder dimension must be 2, i.e. shape = (batch size, max sequence length, ).')

        inputs = inputs.numpy()
        outputs = np.vectorize(self.inverse_vocab_dict.get)(inputs)
        outputs = outputs.tolist()
        outputs = [join_function(o) for o in outputs if o != self.pad_token]
        return outputs

    def to_index(self, inputs: List[str]) -> Tuple[Any, Any]:
        """
        Return corresponding indices of tokens by inputting a list of sentence strings
        
        Args:
            inputs (List[str], length = batch size): list of sentence strings
        
        Returns:
            torch.Tensor, shape = (batch size, max sequence length, ), data_type = torch.long:
                corresponding index of tokens in sentence strings
        """
        outputs = []

        for sent in inputs:
            sent = [self.vocab_dict[t] if t in self.vocab_dict else self.vocab_dict[self.unk_token] for t in
                    self.tokenize(sent)]
            outputs.append(sent)

        perm_tuple = [(c, s) for c, s in
                      sorted(zip(outputs, range(len(outputs))), key=lambda x: len(x[0]), reverse=True)]
        perm_tensors = [torch.Tensor(v[0]) for v in perm_tuple]
        perm_lengths = torch.Tensor([len(sq) for sq in perm_tensors])
        perm_idx = [v[1] for v in perm_tuple]

        pad_tensors = rnn_utils.pad_sequence(perm_tensors,
                                             batch_first=True,
                                             padding_value=self.vocab_dict[self.pad_token])

        desort_idx = list(sorted(range(len(perm_idx)), key=perm_idx.__getitem__))
        desort_tensors = pad_tensors[desort_idx].long()
        desort_lengths = perm_lengths[desort_idx].long()

        return desort_tensors, desort_lengths
