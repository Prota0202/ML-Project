import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm


class CharCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def make_subset(dataset, fraction, seed=42):
    n = len(dataset)
    k = max(1, int(n * fraction))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    return Subset(dataset, idx.tolist())


def train_cnn(model, train_ds, val_ds, *, epochs=3, batch_size=256, lr=1e-3,
              device=None, num_workers=0):
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    dl_tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dl_va = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    history = {"train_loss": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in tqdm(dl_tr, desc=f"epoch {epoch+1}/{epochs}", leave=False):
            xb, yb = xb.to(device), torch.as_tensor(yb).to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history["train_loss"].append(float(np.mean(losses)))

        model.eval()
        accs = []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device)
                yb = torch.as_tensor(yb).to(device)
                pred = model(xb).argmax(1)
                accs.append(float((pred == yb).float().mean().cpu()))
        history["val_acc"].append(float(np.mean(accs)))
        print(f"  epoch {epoch+1}: train_loss={history['train_loss'][-1]:.4f}  val_acc={history['val_acc'][-1]:.4f}")
    return history


@torch.no_grad()
def predict(model, dataset, device=None, batch_size=256):
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device).eval()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    ys, ps = [], []
    for xb, yb in dl:
        xb = xb.to(device)
        pred = model(xb).argmax(1).cpu().numpy()
        ys.append(np.asarray(yb))
        ps.append(pred)
    return np.concatenate(ys), np.concatenate(ps)
