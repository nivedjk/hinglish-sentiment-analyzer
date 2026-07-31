import torch


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")

        print("=" * 60)
        print("GPU DETECTED")
        print(torch.cuda.get_device_name(0))
        print("=" * 60)

        return device

    print("=" * 60)
    print("Using CPU")
    print("=" * 60)

    return torch.device("cpu")