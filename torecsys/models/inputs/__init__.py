import torch.nn as nn

class _Inputs(nn.Module):
    def __init__(self):
        super(_Inputs, self).__init__()

# from audio_inp import AudioInputs
from emb_dict import EmbeddingDict
from field_aware_single_index_emb import FieldAwareSingleIndexEmbedding
from image_inp import ImageInputs
# from image_list_inp import ImageListInputs
from list_index_emb import ListIndexEmbedding
from pretrained_image_inp import PretrainedImageInputs
# from pretrained_text_inp import PretrainedTextInputs
from sequence_index_emb import SequenceIndexEmbedding
from stacked_inp import StackedInputs
from text_inp import TextInputs
# from timestamp_inp import TimestampInputs
from value_inp import ValueInputs
