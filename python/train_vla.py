# python/train_vla.py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
import open_clip
from multiprocessing import freeze_support

from vla_model import NiryoVLA
from action_config import normalize_action, ACTION_DIM

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
DATASET_DIR = "processed_dataset"
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
VAL_SPLIT = 0.1
NUM_WORKERS = 4  # Set to 0 if issues persist
STATE_DIM = 13


# --------------------------------------------------
# DATASET CLASS
# --------------------------------------------------
class NiryoVLADataset(Dataset):
    def __init__(self, dataset_dir, tokenizer):
        self.samples = []
        self.tokenizer = tokenizer

        print(" Scanning processed dataset...")

        for traj in sorted(os.listdir(dataset_dir)):
            traj_path = os.path.join(dataset_dir, traj)
            if not os.path.isdir(traj_path):
                continue

            states_path = os.path.join(traj_path, "states.npy")
            actions_path = os.path.join(traj_path, "actions.npy")
            instruction_path = os.path.join(traj_path, "instruction.txt")
            top_dir = os.path.join(traj_path, "top_images")
            wrist_dir = os.path.join(traj_path, "wrist_images")

            if not (os.path.exists(states_path) and
                    os.path.exists(actions_path) and
                    os.path.exists(instruction_path)):
                print(f"⚠️ Skipping {traj}: Missing required files.")
                continue

            states = np.load(states_path).astype(np.float32)
            actions = np.load(actions_path).astype(np.float32)

            if states.shape[1] != STATE_DIM:
                print(f"⚠️ Skipping {traj}: Invalid state dimension.")
                continue
            if actions.shape[1] != ACTION_DIM:
                print(f"⚠️ Skipping {traj}: Invalid action dimension.")
                continue

            with open(instruction_path, "r") as f:
                instruction = f.read().strip()

            # Tokenize instruction once
            text_tokens = tokenizer([instruction])[0]

            num_samples = len(states)
            for i in range(num_samples):
                top_img_path = os.path.join(top_dir, f"{i:06d}.npy")
                wrist_img_path = os.path.join(wrist_dir, f"{i:06d}.npy")

                if not (os.path.exists(top_img_path) and
                        os.path.exists(wrist_img_path)):
                    continue

                self.samples.append({
                    "top": top_img_path,
                    "wrist": wrist_img_path,
                    "state": states[i],
                    "action": actions[i],
                    "text_tokens": text_tokens
                })

            print(f"✅ Loaded {traj} with {num_samples} samples.")

        print(f"📊 Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        top = np.load(sample["top"])
        wrist = np.load(sample["wrist"])

        # Convert to tensor (C, H, W)
        top = torch.from_numpy(top).permute(2, 0, 1).float()
        wrist = torch.from_numpy(wrist).permute(2, 0, 1).float()

        state = torch.from_numpy(sample["state"]).float()

        # Normalize action to [-1, 1]
        action = normalize_action(sample["action"])
        action = torch.from_numpy(action).float()

        text_tokens = sample["text_tokens"]

        return top, wrist, state, text_tokens, action


# --------------------------------------------------
# MAIN TRAINING FUNCTION
# --------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    dataset = NiryoVLADataset(DATASET_DIR, tokenizer)

    if len(dataset) == 0:
        raise RuntimeError(
            "❌ No valid samples found in the dataset. "
            "Ensure actions.npy files have the correct 6D format."
        )

    # Train/Validation split
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # Model, optimizer, and loss
    model = NiryoVLA(state_dim=STATE_DIM, action_dim=ACTION_DIM, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        # ------------------ Training ------------------
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for top, wrist, state, text_tokens, action in loop:
            top = top.to(device, non_blocking=True)
            wrist = wrist.to(device, non_blocking=True)
            state = state.to(device, non_blocking=True)
            text_tokens = text_tokens.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(top, wrist, state, text_tokens=text_tokens)
            loss = criterion(pred, action)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)

        # ------------------ Validation ------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for top, wrist, state, text_tokens, action in val_loader:
                top = top.to(device, non_blocking=True)
                wrist = wrist.to(device, non_blocking=True)
                state = state.to(device, non_blocking=True)
                text_tokens = text_tokens.to(device, non_blocking=True)
                action = action.to(device, non_blocking=True)

                pred = model(top, wrist, state, text_tokens=text_tokens)
                loss = criterion(pred, action)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/niryo_vla.pth")
            print("💾 Best model saved.")

    print("✅ Model training complete!")


# --------------------------------------------------
# ENTRY POINT (Required for Windows)
# --------------------------------------------------
if __name__ == "__main__":
    freeze_support()
    main()