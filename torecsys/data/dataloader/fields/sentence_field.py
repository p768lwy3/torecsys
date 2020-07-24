import warnings
from collections import Counter
from typing import Callable, List, Tuple

import torch
import torch.nn.utils.rnn as rnn_utils

from torecsys.utils.decorator import to_be_tested


@to_be_tested
class SentenceField(object):
    r"""SentenceField is a field for collate_fn of DataLoader to parse Sentence String"""

    def __init__(self,
                 tokenize: Callable[[str], list] = lambda x: x.split(),
                 threshold: int = 0,
                 pad_token: Tuple[str, int] = ("<pad>", 0),
                 unk_token: Tuple[str, int] = ("<unk>", 1)):
        r"""initialize sentence field
        
        Args:
            tokenize (Callable[[str], list], optional): function to split input sentences to list of string. Defaults to lambda x:x.split().
            threshold (int, optional): threshold for dropping tokens. Defaults to 0.
            pad_token (Tuple[str, int], optional): padding token and corresponding index. Defaults to ("<pad>", 0).
            unk_token (Tuple[str, int], optional): unknown token and corresponding index. Defaults to ("<unk>", 1).
        """
        # raise warning to show the method have not been tested
        warnings.warn("Warning: the method haven't been tested.")

        # initialize variables of object
        self.tokenize = tokenize
        self.threshold = threshold

        # vocab_dict : key = string of token, value = index of token
        self.vocab_dict = dict([pad_token, unk_token])
        self.pad_token = pad_token[0]
        self.unk_token = unk_token[0]

        # vocab_count: key = string of token, value = word counts
        self.vocab_count = dict()

        # is_initialized
        self.is_initialized = False

    def __len__(self) -> int:
        r"""Return number of vocabulary in the field
        
        Returns:
            int: number of vocabulary in the field
        """
        return len(self.vocab_dict)

    def _is_initialized(self) -> bool:
        r"""Return True if the filed have been initialized, else Return False
        
        Returns:
            bool: boolean flag to show whether the field is initialized
        """
        return self.is_initialized

    def build_vocab(self, dataset: List[str]):
        r"""Build vocabulary with input dataset
        
        Args:
            dataset (List[str]): list of sentence strings for building the vocabulary
        """
        # get max id of the vocab_dict at this stage
        max_id = max(list(self.vocab_dict.values()))

        # count vocab in dataset
        counts = dict(Counter(tok for sent in dataset for tok in self.tokenize(sent)))

        # append to vocab_dict and vocab_count if not exist and counts is larger than threshold, 
        # else update vocab_count
        appended = 0
        updated = 0
        for k, v in counts.items():
            if k not in self.vocab_dict and v > self.threshold:
                # append to vocab_dict and vocab_count
                max_id += 1
                self.vocab_count[k] = v
                self.vocab_dict[k] = max_id
                appended += 1
            elif k in self.vocab_count:
                # update the value of vocab_count
                self.vocab_count[k] += v
                updated += 1
            else:
                continue

        print("total number of vocabulary added : %s." % appended)
        print("total number of vocabulary updated : %s." % updated)

        # update inverse vocab dict
        self._update_inverse_dict()

        # set self.is_initialized to True
        self.is_initialized = True

    def _update_inverse_dict(self):
        r"""to build inverse dictionary where the key is index and the value is string token
        """
        # inverse_vocab_dict: key = index of token, value = string of token
        self.inverse_vocab_dict = {v: k for k, v in self.vocab_dict.items()}

    def to_index(self, inputs: List[str]) -> torch.Tensor:
        r"""Return corresponding indices of tokens by inputting a list of sentence strings
        
        Args:
            inputs (List[str], length = batch size): list of sentence strings
        
        Returns:
            torch.Tensor, shape = (batch size, max sequence length, ), dtype = torch.long: corresponding index of tokens in sentence strings
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

    def from_index(self,
                   inputs: torch.Tensor,
                   join_function: Callable[[List[str]], str] = "".join) -> List[str]:
        r"""Return list of sentence string from a torch.Tensor of corresponding indicies
        
        Args:
            inputs (torch.Tensor, shape = (batch size, max sequence length, ), dtype = torch.long): corresponding index of tokens in sentence strings
            join_function (Callable[[List[str]], str], optional): function to join the token string into a sentence. Defaults to "".join.
        
        Raises:
            ValueError: when the inputs dimension is not equal to 2, i.e. (batch size, max sequence length, )
        
        Returns:
            List[str]: list of sentence strings
        """
        # check whether dimension of inputs is 2
        if inputs.dim() != 2:
            raise ValueError("inputs dimension must be 2, i.e. shape = (batch size, max sequence length, ).")

        # convert tensor to numpy, then take vectorized mapping
        inputs = inputs.numpy()
        outputs = np.vectorize(self.inverse_vocab_dict.get)(inputs)
        outputs = outputs.tolist()
        outputs = [join_function(o) for o in outputs if o != self.pad_token]

        return outputs
