import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from CIFAR10_ORAT import run_orat_training


class NpyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_args():
    parser = argparse.ArgumentParser(description="Landseer wrapper for ORAT")
    parser.add_argument("--input-dir", default="/data", help="Input directory mounted by Landseer")
    parser.add_argument("--output", default="/output", help="Output directory mounted by Landseer")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--epsilon", type=float, default=0.0078)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--step_size", type=float, default=0.0078 / 4)
    parser.add_argument("--k", type=int, default=50000)
    parser.add_argument("--m", type=int, default=500)
    parser.add_argument("--aorr", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", type=str, default="")

    return parser.parse_args()


def main():
    args = get_args()

    os.makedirs(args.output, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x_train = np.load(os.path.join(args.input_dir, "data.npy")).astype(np.float32)
    y_train = np.load(os.path.join(args.input_dir, "labels.npy")).astype(np.int64)
    x_test = np.load(os.path.join(args.input_dir, "test_data.npy")).astype(np.float32)
    y_test = np.load(os.path.join(args.input_dir, "test_labels.npy")).astype(np.int64)

    train_dataset = NpyDataset(x_train, y_train)
    test_dataset = NpyDataset(x_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    from config_model import config
    model = config().to(device)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = run_orat_training(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        epsilon=args.epsilon,
        num_steps=args.num_steps,
        step_size=args.step_size,
        k=args.k,
        m=args.m,
        aorr=args.aorr,
        out_dir=args.output,
        resume=args.resume,
    )

    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), os.path.join(args.output, "model.pt"))

    print(f"Saved trained model to {os.path.join(args.output, 'model.pt')}")


if __name__ == "__main__":
    main()
