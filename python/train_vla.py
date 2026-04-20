import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
from multiprocessing import freeze_support

from vla_model import NiryoVLA
from action_config import normalize_action, ACTION_DIM

DATASET_DIR = "processed_dataset"
BATCH_SIZE  = 16
EPOCHS      = 30
LR          = 1e-4
VAL_SPLIT   = 0.1
NUM_WORKERS = 4
STATE_DIM   = 13
GOAL_DIM    = 6


class NiryoVLADataset(Dataset):
    """
    CHANGED: loads goal_pose.npy instead of instruction.txt + tokenizer.

    Each sample:
        top        [3, 224, 224]  top camera image
        wrist      [3, 224, 224]  wrist camera image
        state      [13]           robot state
        goal_pose  [6]            normalized goal EE pose (from trajectory end)
        action     [6]            normalized absolute EE target for this step
    """

    def __init__(self, dataset_dir):
        self.samples = []
        print("Scanning processed dataset...")

        for traj in sorted(os.listdir(dataset_dir)):
            traj_path = os.path.join(dataset_dir, traj)
            if not os.path.isdir(traj_path):
                continue

            states_path    = os.path.join(traj_path, "states.npy")
            actions_path   = os.path.join(traj_path, "actions.npy")
            goal_pose_path = os.path.join(traj_path, "goal_pose.npy")   # NEW
            top_dir        = os.path.join(traj_path, "top_images")
            wrist_dir      = os.path.join(traj_path, "wrist_images")

            if not all(os.path.exists(p) for p in
                       [states_path, actions_path, goal_pose_path]):
                print(f"  Skipping {traj}: missing files.")
                continue

            states    = np.load(states_path).astype(np.float32)
            actions   = np.load(actions_path).astype(np.float32)
            goal_pose = np.load(goal_pose_path).astype(np.float32)   # [6]

            if states.shape[1] != STATE_DIM:
                print(f"  Skipping {traj}: bad state dim {states.shape[1]}")
                continue
            if actions.shape[1] != ACTION_DIM:
                print(f"  Skipping {traj}: bad action dim {actions.shape[1]}")
                continue

            # Normalize goal_pose to [-1, 1] using same bounds as actions
            goal_pose_norm = normalize_action(goal_pose)

            for i in range(len(states)):
                top_path   = os.path.join(top_dir,   f"{i:06d}.npy")
                wrist_path = os.path.join(wrist_dir, f"{i:06d}.npy")

                if not os.path.exists(top_path) or not os.path.exists(wrist_path):
                    continue

                self.samples.append({
                    "top":           top_path,
                    "wrist":         wrist_path,
                    "state":         states[i],
                    "goal_pose":     goal_pose_norm,   # same for all steps
                    "action":        normalize_action(actions[i]),
                })

            print(f"  Loaded {traj}: {len(states)} samples, "
                  f"goal=({goal_pose[0]:.3f},{goal_pose[1]:.3f},{goal_pose[2]:.3f})")

        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        top   = torch.from_numpy(np.load(s["top"])).permute(2, 0, 1).float()
        wrist = torch.from_numpy(np.load(s["wrist"])).permute(2, 0, 1).float()
        state     = torch.from_numpy(s["state"]).float()
        goal_pose = torch.from_numpy(s["goal_pose"]).float()
        action    = torch.from_numpy(s["action"]).float()

        return top, wrist, state, goal_pose, action


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = NiryoVLADataset(DATASET_DIR)

    if len(dataset) == 0:
        raise RuntimeError("No valid samples found. Run pre_process.py first.")

    val_size   = max(1, int(len(dataset) * VAL_SPLIT))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model     = NiryoVLA(state_dim=STATE_DIM, action_dim=ACTION_DIM, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        # ── Training ──────────────────────────────────────
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]")

        for top, wrist, state, goal_pose, action in loop:
            top       = top.to(device, non_blocking=True)
            wrist     = wrist.to(device, non_blocking=True)
            state     = state.to(device, non_blocking=True)
            goal_pose = goal_pose.to(device, non_blocking=True)
            action    = action.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(top, wrist, state, goal_pose)   # CHANGED signature
            loss = criterion(pred, action)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.5f}")

        train_loss /= len(train_loader)

        # ── Validation ────────────────────────────────────
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for top, wrist, state, goal_pose, action in val_loader:
                top       = top.to(device, non_blocking=True)
                wrist     = wrist.to(device, non_blocking=True)
                state     = state.to(device, non_blocking=True)
                goal_pose = goal_pose.to(device, non_blocking=True)
                action    = action.to(device, non_blocking=True)

                pred = model(top, wrist, state, goal_pose)
                val_loss += criterion(pred, action).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"  Train: {train_loss:.6f}  Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/niryo_vla.pth")
            print("  Saved best model.")

    print("Training complete.")


if __name__ == "__main__":
    freeze_support()
    main()