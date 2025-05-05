from typing import Any

import torch


def move_dict_of_tensors_to_device(dictionary: dict[Any, torch.Tensor], device: str):
    """
    Moves all tensors in a dictionary to a specific device.
    :param dictionary:
    :param device:
    :return:
    """
    for key in dictionary:
        dictionary[key] = dictionary[key].to(device)


