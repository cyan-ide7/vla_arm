import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from action_config import ACTION_DIM

# --------------------------------------------------
# CHANGED ARCHITECTURE SUMMARY
#
# REMOVED:  CLIP text encoder (512d, hard to train,
#           requires 50k+ demos to ground language)
#
# ADDED:    goal_pose encoder (6D → 32d MLP)
#           Claude API handles the language grounding
#           externally. This model only needs to learn:
#           (images, state, where_to_go) → next_ee_pose
#
# Fusion input:
#   top_feat    256
#   wrist_feat  256
#   state_feat   64
#   goal_feat    32
#   TOTAL       608  (was 704)
# --------------------------------------------------

GOAL_POSE_DIM = 6    # [x, y, z, pitch, yaw, gripper]


class NiryoVLA(nn.Module):
    """
    Vision-Action model conditioned on a goal end-effector pose.

    The goal pose comes from claude_parser.py at inference time,
    and from goal_pose.npy (extracted from trajectory end state)
    at training time.

    Inputs:
        top_img    [B, 3, 224, 224]
        wrist_img  [B, 3, 224, 224]
        state      [B, 13]
        goal_pose  [B, 6]   <- replaces text tokens

    Output:
        action     [B, 6]   normalized EE pose in [-1, 1]
    """

    def __init__(self, state_dim=13, action_dim=ACTION_DIM, device=None):
        super(NiryoVLA, self).__init__()

        self.device = device if device is not None else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Vision encoders (unchanged)
        self.top_encoder = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.top_encoder.fc = nn.Linear(512, 256)

        self.wrist_encoder = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.wrist_encoder.fc = nn.Linear(512, 256)

        # State encoder (unchanged)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # NEW: goal pose encoder — replaces the 512d CLIP text encoder.
        # 6D normalized goal pose → 32d feature.
        # Small on purpose: goal pose is already very structured data,
        # it doesn't need a large embedding.
        self.goal_encoder = nn.Sequential(
            nn.Linear(GOAL_POSE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Fusion: 256 + 256 + 64 + 32 = 608
        self.fusion = nn.Sequential(
            nn.Linear(608, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

        self.to(self.device)

    def forward(self, top_img, wrist_img, state, goal_pose):
        """
        Args:
            top_img   [B, 3, 224, 224]
            wrist_img [B, 3, 224, 224]
            state     [B, 13]
            goal_pose [B, 6]   normalized to [-1, 1] by action_config

        Returns:
            action    [B, 6]  normalized EE pose in [-1, 1]
        """
        top_img   = top_img.to(self.device)
        wrist_img = wrist_img.to(self.device)
        state     = state.to(self.device)
        goal_pose = goal_pose.to(self.device)

        top_feat   = self.top_encoder(top_img)       # [B, 256]
        wrist_feat = self.wrist_encoder(wrist_img)   # [B, 256]
        state_feat = self.state_encoder(state)       # [B,  64]
        goal_feat  = self.goal_encoder(goal_pose)    # [B,  32]

        fused  = torch.cat([top_feat, wrist_feat, state_feat, goal_feat], dim=1)
        action = self.fusion(fused)                  # [B, 6]

        return action