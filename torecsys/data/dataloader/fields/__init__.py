"""
torecsys.data.dataloader.fields is a sub model to initialize fields for collate_fn of
torecsys.utils.data.DataLoader to parse Dataset into batched data
"""

from abc import ABC


class Field(ABC):
    pass


from torecsys.data.dataloader.fields.index_field import IndexField
from torecsys.data.dataloader.fields.sentence_field import SentenceField
