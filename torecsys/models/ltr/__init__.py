"""
torecsys.models.ltr is a sub model of the implementations of the whole models of Learning-to-Rank algorithm
"""

from torecsys.models import BaseModel


class LtrBaseModel(BaseModel):
    def __init__(self):
        super().__init__()


class ReRankingModel(LtrBaseModel):
    def __init__(self):
        super().__init__()


from torecsys.models.ltr.personalized_reranking import PersonalizedReRankingModel
