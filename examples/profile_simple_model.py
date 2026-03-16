import torch


def main() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 256),
        torch.nn.ReLU(),
    )
    x = torch.randn(64, 512)
    for _ in range(20):
        _ = model(x)


if __name__ == "__main__":
    main()
