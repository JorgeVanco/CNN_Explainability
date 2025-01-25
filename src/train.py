# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm
from typing import Final

# own modules
from src.utils import load_data, save_model, train_step, val_step
from src.models import Cifar10ConvModel


# static variables
DATA_PATH: Final[str] = "data"
NUM_CLASSES: Final[int] = 10

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main() -> None:
    """
    This function is the main program for training.
    """
    print("Using: ", device)

    # hyperparameters
    epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 32

    # empty nohup file
    open("nohup.out", "w").close()

    # load data
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, _ = load_data(DATA_PATH, batch_size=batch_size)

    # define name and writer
    name: str = (
        "scheduler_run"  # f"model_lr_{lr}_hs_{hidden_sizes}_{batch_size}_{epochs}"
    )
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # define model
    model: torch.nn.Module = Cifar10ConvModel(0.1).to(device)

    # define loss and optimizer
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # train loop
    for epoch in tqdm(range(epochs)):
        # call train step
        train_step(model, train_data, loss, optimizer, writer, epoch, device)
        scheduler.step()
        # call val step
        val_step(model, val_data, loss, writer, epoch, device)

    # save model
    save_model(model, name)

    return None


if __name__ == "__main__":
    main()
