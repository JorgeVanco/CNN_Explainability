import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

cifar_classes = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def freeze_model(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False


def compute_gradients_input(
    model: nn.Module, inputs: torch.Tensor, device: torch.device
) -> torch.Tensor:
    inputs.requires_grad = True  # force grad
    output = model(inputs.to(device))

    # Compute gradients
    max_output, max_indices = output.max(dim=-1)
    max_output.sum().backward()

    return max_indices


def show_saliency_map_grid(
    inputs, grads, targets, max_indices, ncol=None
) -> plt.Figure:
    if ncol is None:
        ncol = 2

    # Normalize gradients
    no_channel_grads, _ = torch.max(grads.abs(), dim=1)
    no_channel_grads = no_channel_grads.unsqueeze(1).abs()
    max_vals = no_channel_grads.amax(dim=(1, 2, 3), keepdim=True)
    min_vals = no_channel_grads.amin(dim=(1, 2, 3), keepdim=True)

    normalized_grads = (no_channel_grads - min_vals) / (max_vals - min_vals)

    # Create subplots
    num_images = inputs.size(0)
    fig, axes = plt.subplots(num_images // ncol, ncol * 2, figsize=(10, 10))

    invTrans = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.247, 1 / 0.243, 1 / 0.261]
            ),
            transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1.0, 1.0, 1.0]),
        ]
    )

    for i in range(num_images):
        row = i // ncol
        col = (i % ncol) * 2

        # Plot input image
        img = invTrans(inputs[i])
        npimg = img.numpy()
        axes[row, col].imshow(np.transpose(npimg, (1, 2, 0)))
        axes[row, col].set_title(f"Class: {cifar_classes[targets[i].item()]}")
        axes[row, col].axis("off")

        # Plot gradient image
        grad_img = normalized_grads[i].squeeze().cpu().numpy()
        axes[row, col + 1].imshow(grad_img, cmap="gray")
        axes[row, col + 1].set_title(f"Pred: {cifar_classes[max_indices[i].item()]}")
        axes[row, col + 1].axis("off")

    plt.suptitle("Saliency Maps")
    plt.tight_layout()
    return fig


def calc_mean_image(dataloader) -> torch.Tensor:
    for i, (input2, target) in enumerate(dataloader):
        if i == 0:
            mean_image = input2.mean(dim=0)
        else:
            mean_image += input2.mean(dim=0)
    mean_image /= len(dataloader)
    return mean_image


def show_class_model_visualization_grid(results, ncol=None) -> plt.Figure:
    if ncol is None:
        ncol = 5

    results = (results - results.min()) / (results.max() - results.min())
    # Create subplots
    num_images = results.size(0)
    fig, axes = plt.subplots(num_images // ncol, ncol, figsize=(8, 4))

    for i in range(num_images):
        row = i // ncol
        col = i % ncol

        # Plot input image
        img = results[i]
        npimg = img.numpy()
        axes[row, col].imshow(np.transpose(npimg, (1, 2, 0)))
        axes[row, col].set_title(f"Class: {cifar_classes[i]}")
        axes[row, col].axis("off")
    plt.suptitle("Class model visualization")
    plt.tight_layout()
    return fig


def save_pdf(figures, path) -> None:
    with PdfPages(path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
        plt.close()
