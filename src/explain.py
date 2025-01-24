# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.jit import RecursiveScriptModule

# other libraries
from tqdm.auto import tqdm
from typing import Final

# own modules
from src.utils import test_step, load_data, show_saliency_map_grid

# static variables
DATA_PATH: Final[str] = "data"
batch_size: int = 8

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main() -> None:
    train_data, val_data, test_data = load_data(DATA_PATH, batch_size=batch_size)
    name: str = "run5"

    # define model
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt").to(device)

    # call test step and evaluate accuracy
    accuracy: float = test_step(model, test_data, device)
    print(f"Model test accuracy: {accuracy}")

    # Freeze parameters
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    # get a batch
    input, target = next(iter(train_data))

    for tensor in input:
        tensor.grad = None
        tensor.requires_grad = True  # force grad
    input.requires_grad = True  # force grad
    output = model(input.to(device))

    # Compute gradients
    max_output, max_indices = output.max(dim=-1)
    max_output.sum().backward()

    show_saliency_map_grid(input.detach().cpu(), input.grad, target, max_indices)


if __name__ == "__main__":
    main()
