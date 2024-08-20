import datetime
import logging
import os
import sys
import time

import numpy as np
import transformers
import wandb
from matplotlib import pyplot as plt
from transformers import set_seed

from loralib import *
from utils.architecture import *
from utils.optimizer import *

logger = logging.getLogger(__name__)


def get_arch(model_name_or_path, model):
    if model_name_or_path == 'arampacha/roberta-tiny':
        hidden_layers = 4
    elif model_name_or_path == 'roberta-base' or model_name_or_path == 'microsoft/deberta-v3-base':
        hidden_layers = 12
    elif model_name_or_path == 'roberta-large':
        hidden_layers = 24
    elif model_name_or_path == 'microsoft/deberta-v2-xxlarge':
        hidden_layers = 48
    else:
        raise NotImplementedError()

    return BiDoRAArchitecture(model, hidden_layers)


def check_dir(folder_path):
    if not os.path.exists(folder_path):
        print(f'created path {folder_path}')
        os.makedirs(folder_path)
    else:
        print(f'path {folder_path} already exists')


def compute_direction_regularization(model, regu_weight=0.1):
    '''
    Regularize on the direction matrix, try to make the directions orthogonoal to each other
    '''
    regu_loss, num_param = 0., 0
    for module_name, module in model.named_modules():
        if isinstance(module, BiDoRALinear):
            D = module.v_ft()
            D_ = D.T @ D
            I = torch.eye(len(D_), device=D_.device)
            regu_loss += torch.norm(D_ - I, p="fro")
            # regu_loss += torch.linalg.matrix_norm(D_ - I, ord=2)
            num_param += 1

    return regu_weight * regu_loss / num_param


def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    return tensors.cpu().detach()


def nested_truncate(tensors, limit):
    "Truncate `tensors` at `limit` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_truncate(t, limit) for t in tensors)
    return tensors[:limit]


def set_configuration(model_training_args):
    torch.use_deterministic_algorithms(model_training_args.use_deterministic_algorithms)
    logger.info("use_deterministic_algorithms: " + str(torch.are_deterministic_algorithms_enabled()))
    model_training_args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('output.log')],
        level=logging.INFO
    )
    logger.setLevel(logging.WARN)

    logger.warning(
        f"Process rank: {model_training_args.local_rank}, device: {model_training_args.device}, n_gpu: {model_training_args.n_gpu}"
        + f"distributed training: {bool(model_training_args.local_rank != -1)}, 16-bits training: {model_training_args.fp16}"
    )
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {model_training_args}")

    set_seed(model_training_args.seed)


def numpy_pad_and_concatenate(array1, array2, padding_index=-100):
    """Concatenates `array1` and `array2` on first axis, applying padding on the second if necessary."""
    if len(array1.shape) == 1 or array1.shape[1] == array2.shape[1]:
        return np.concatenate((array1, array2), dim=0)

    # Let's figure out the new shape
    new_shape = (array1.shape[0] + array2.shape[0], max(array1.shape[1], array2.shape[1])) + array1.shape[2:]

    # Now let's fill the result tensor
    result = np.full_like(array1, padding_index, shape=new_shape)
    result[: array1.shape[0], : array1.shape[1]] = array1
    result[array1.shape[0]:, : array2.shape[1]] = array2
    return result


def torch_pad_and_concatenate(tensor1, tensor2, padding_index=-100):
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""
    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return torch.cat((tensor1, tensor2), dim=0)

    # Let's figure out the new shape
    new_shape = (tensor1.shape[0] + tensor2.shape[0], max(tensor1.shape[1], tensor2.shape[1])) + tensor1.shape[2:]

    # Now let's fill the result tensor
    result = tensor1.new_full(new_shape, padding_index)
    result[: tensor1.shape[0], : tensor1.shape[1]] = tensor1
    result[tensor1.shape[0]:, : tensor2.shape[1]] = tensor2
    return result


def nested_numpify(tensors):
    "Numpify `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    return tensors.cpu().numpy()


def nested_concat(tensors, new_tensors, padding_index=-100):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples of tensors.
    """
    assert type(tensors) == type(
        new_tensors
    ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, padding_index=padding_index) for t, n in zip(tensors, new_tensors))
    elif isinstance(tensors, torch.Tensor):
        return torch_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    elif isinstance(tensors, np.ndarray):
        return numpy_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    else:
        raise TypeError(f"Unsupported type for concatenation: got {type(tensors)}")


class AverageMeter(object):
    """Computes and stores the average and current value
         Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    ...
