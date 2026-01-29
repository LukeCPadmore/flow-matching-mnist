import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

default_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Pad(2, padding_mode="constant"),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

TRANSFORMS = {"default": default_transform, "none": transforms.ToTensor()}


def build_transform(name: str):
    if name == "default":
        return default_transform
    if name == "none":
        return transforms.ToTensor()
    raise ValueError(f"Unknown transform preset: {name}")


def create_mnist_train_val_loaders(
    batch_size: int = 64,
    data_path: str = "/home/luke-padmore/Source/flow-matching-mnist/data",
    transform: str = "default",
    num_workers=4,
    shuffle=True,
) -> tuple[DataLoader, DataLoader]:
    transform = TRANSFORMS.get(transform, default_transform)
    trainset = torchvision.datasets.MNIST(
        root=data_path, train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    val_set = torchvision.datasets.MNIST(
        root=data_path,
        train=False,
        download=True,
        transform=transform,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return train_loader, val_loader
