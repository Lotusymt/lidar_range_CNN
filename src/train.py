import os
import sys
import torch
from torch.utils.data import DataLoader, ConcatDataset   # ConcatDataset = combine multiple sequences
from torch import nn, optim
from tqdm import tqdm
import multiprocessing

from dataset import KittiPairDataset
from model import PairRangeCNN

if sys.platform == 'win32':
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except (AttributeError, RuntimeError):
        pass


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.BCELoss()
    running_loss = 0.0
    correct = 0
    total = 0

    for pairs, labels in tqdm(loader):
        pairs = pairs.to(device)                 # (B, 2, H, W)
        labels = labels.to(device).unsqueeze(1)  # (B,) -> (B, 1)

        optimizer.zero_grad()
        probs = model(pairs)                     # (B, 1), sigmoid output
        loss = criterion(probs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * pairs.size(0)
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.numel()

    return running_loss / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    criterion = nn.BCELoss()
    running_loss = 0.0
    correct = 0
    total = 0

    for pairs, labels in loader:
        pairs = pairs.to(device)
        labels = labels.to(device).unsqueeze(1)

        probs = model(pairs)
        loss = criterion(probs, labels)

        running_loss += loss.item() * pairs.size(0)
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.numel()

    return running_loss / total, correct / total


def main():
    # Set multiprocessing start method for Windows compatibility
    if sys.platform == 'win32':
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
    
    # ----- 1. Choose which sequences for train / test -----
    # 00–10 have poses; you can change these lists any time.
    train_seqs = ["00", "01", "02", "03", "04", "05", "06", "07"]
    test_seqs  = ["08", "09", "10"]

    # ----- 2. Build one dataset per sequence -----
    # num_pairs = how many random pairs we draw *per epoch* from that sequence
    train_dsets = [KittiPairDataset(seq=s, num_pairs=4000) for s in train_seqs]
    test_dsets  = [KittiPairDataset(seq=s, num_pairs=1000) for s in test_seqs]

    # Combine all training sequences into one big dataset
    train_dataset = ConcatDataset(train_dsets)
    test_dataset  = ConcatDataset(test_dsets)

    # DataLoader loads only one batch at a time (not the whole dataset into RAM)
    # On Windows, use 'spawn' multiprocessing context to avoid shared memory issues
    requested_workers = int(os.getenv("NUM_WORKERS", "4"))
    num_workers = requested_workers
    if sys.platform == 'win32':
        num_workers = min(requested_workers, 2)

    mp_context = multiprocessing.get_context('spawn') if sys.platform == 'win32' else None
    prefetch = int(os.getenv("PREFETCH_FACTOR", "1")) if num_workers > 0 else None

    loader_common = dict(num_workers=num_workers)
    if mp_context is not None:
        loader_common["multiprocessing_context"] = mp_context
    if prefetch is not None:
        loader_common["prefetch_factor"] = prefetch
        loader_common["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, **loader_common)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, **loader_common)

    # ----- 3. Model + optimizer -----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = PairRangeCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ----- 4. Training loop -----
    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, test_loader, device)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    # ----- 5. Save model -----
    torch.save(model.state_dict(), "pair_range_cnn_kitti_00_10.pth")


if __name__ == "__main__":
    main()
