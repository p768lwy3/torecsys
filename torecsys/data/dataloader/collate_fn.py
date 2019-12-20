from io import BytesIO
from PIL import Image
import os
import requests
import torch
import torch.nn.utils.rnn as rnn_utils
import torchvision
import torchvision.transforms as transforms
from typing import Dict, List, Tuple
import warnings

__field_type__ = ["values", "single_index", "list_index", "sequence_index", "image_dir", "image_url", "sentence"]

def dict_collate_fn(batch_data: List[Dict[str, list]],
                    schema    : Dict[str, Tuple[str, str]],
                    device    : str = "cpu",
                    **kwargs) -> Dict[str, torch.Tensor]:
    r"""Collate function to transform data from dataset to batch in dataloader.
    
    Args:
        batch_data (List[Dict[str, list]]): List of dictionary, where its keys are inputs' field names
        schema (Dict[str, Tuple[str, str]]): Schema to transform the raw inputs to the form of inputting to model
        device (str, optional): Device of torch. 
            Defaults to "cpu".
    
    Raises:
        warning: when output type defined in schema does not exist.
    
    Returns:
        Dict[str, T]: Dictionary of input of model, where its keys are the name of arguments of model, and the values are \
            tensors which their device types should be equal to the model.
    """
    # initialize outputs dictionary to store tensors of fields
    outputs = dict()

    # loop through schema for each input's field
    for inp_key, (out_key, out_type) in schema.items():
        # get input's values by using input name as a key
        inp_value = [data[inp_key] for data in batch_data]

        if out_type == "values":
            # return tensor with dtype = torch.float()
            outputs[out_key] = torch.Tensor(inp_value).to(device)
        
        elif out_type == "single_index":
            # return tensor with dtype = torch.long()
            outputs[out_key] = torch.Tensor(inp_value).long().to(device)
        
        elif out_type == "list_index":
            # to get the descending sorted list and their perm index
            perm_tuple = [(c, s) for c, s in sorted(zip(inp_value, range(len(inp_value))), key=lambda x: len(x[0]), reverse=True)]

            # to convert lists in the list to tensor for rnn_utils.pad_sequence
            perm_tensors = [torch.Tensor(v[0]) for v in perm_tuple]
            perm_idx = [v[1] for v in perm_tuple]

            # pad the list of tensors
            pad_tensors = rnn_utils.pad_sequence(perm_tensors, batch_first=True, padding_value=0)

            # to get the desort index
            desort_idx = list(sorted(range(len(perm_idx)), key=perm_idx.__getitem__))
            desort_tensors = pad_tensors[desort_idx]

            # return tensor with dtype = torch.long()
            outputs[out_key] = desort_tensors.long().to(device)
        
        elif out_type == "sequence_index":
            warnings.warn("this is not yet checked")
            
            # to get the descending sorted list and their perm index
            perm_tuple = [(c, s) for c, s in sorted(zip(inp_value, range(len(inp_value))), key=lambda x: len(x[0]), reverse=True)]

            # to convert lists in the list to tensor for rnn_utils.pad_sequence
            perm_tensors = [torch.Tensor(v[0]) for v in perm_tuple]
            perm_lengths = torch.Tensor([len(sq) for sq in perm_tensors])
            perm_idx = [v[1] for v in perm_tuple]

            # pad the list of tensors
            pad_tensors = rnn_utils.pad_sequence(perm_tensors, batch_first=True, padding_value=0)

            # to get the desort index
            desort_idx = list(sorted(range(len(perm_idx)), key=perm_idx.__getitem__))
            desort_tensors = pad_tensors[desort_idx].long().to(device)
            desort_lengths = perm_lengths[desort_idx].long().to(device)

            # save desort_tensors to outputs dictionary
            outputs[out_type] = desort_tensors
            outputs[out_type + "_length"] = desort_lengths
        
        elif out_type == "image_dir":
            warnings.warn("this is not yet checked")

            # get root dir from kwargs
            root_dir = kwargs.get("image_rootdir", {}).get(out_key, os.getcwd())

            # get transform
            img_transforms = kwargs.get("image_transforms", {}).get(out_key, transforms.ToTensor())
            
            # join root directory with file path
            img_files = [os.path.join(root_dir, img_path[0]) for img_path in inp_value]

            # read image with skimage.io
            img_files = [Image.open(img_file) for img_file in img_files]

            # apply transform with torch.transforms and stack them into a tensor
            img_tensors = [img_transforms(img_file) for img_file in img_files]
            img_tensors = torch.stack(img_tensors)

            # save to outputs dictionary where the shape = (batch size, dimensionality of color, height, width)
            outputs[out_type] = img_tensors.to(device)
        
        elif out_type == "image_url":
            warnings.warn("this is not yet checked")

            # get root url from kwargs
            root_url = kwargs.get("image_rooturl", {}).get(out_key, "")

            # get transform
            img_transforms = kwargs.get("image_transforms", {}).get(out_key, transforms.ToTensor())

            # join root url with file path
            img_urls = [root_url + img_url[0] for img_url in inp_value]
            
            # read image with skimage.io
            img_files = [Image.open(BytesIO(requests.get(img_url).content)) for img_url in img_urls]
            
            # apply transform with torch.transforms and stack them into a tensor
            img_tensors = [img_transforms(img_file) for img_file in img_files]
            img_tensors = torch.stack(img_tensors)

            # save to outputs dictionary where the shape = (batch size, dimensionality of color, height, width)
            outputs[field_name] = img_tensors.to(device)
        
        elif out_type == "sentence":
            warnings.warn("this is not yet checked")

            # get vocab field which is initialized
            sentence_field = kwargs.get("sentence_fields", {}).get(out_key)

            # apply sentence_field.to_index
            sent_tensors, sent_lengths = sentence_field.to_index(inp_value)

            # save to outputs dictionary
            outputs[field_name] = sent_tensors.to(device)
            outputs[field_name + "_length"] = sent_lengths.to(device)
        
        else:
            # raise warning if the output type is not found
            warnings.warn("output type : %s doesn't exist." % out_type)
        
    return outputs
