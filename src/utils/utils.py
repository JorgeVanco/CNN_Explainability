import torch
import os
import numpy as np
import random


def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model in the 'models' folder as a torch.jit.
    It should create the 'models' if it doesn't already exist.

    Args:
        model: pytorch model.
        name: name of the model (without the extension, e.g. name.pt).
    """

    # create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    # save scripted model
    model_scripted = torch.jit.script(model.cpu())
    model_scripted.save(f"models/{name}.pt")

    return None


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior

    Args:
        seed: seed number to fix radomness
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    This function computes the accuracy.

    Args:
        predictions: predictions tensor. Dimensions:
            [batch, num classes] or [batch].
        targets: targets tensor. Dimensions: [batch, 1] or [batch].

    Returns:
        the accuracy in a tensor of a single element.
    """

    predictions = torch.argmax(predictions, dim=-1)
    return torch.sum(predictions == targets) / predictions.shape[0]
