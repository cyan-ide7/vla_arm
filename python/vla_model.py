# python/vla_model.py
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
import open_clip

from action_config import ACTION_DIM


class NiryoVLA(nn.Module):
    """
    Vision-Language-Action (VLA) model for the Niryo robotic arm.

    Inputs:
        - Top camera image:  [B, 3, 224, 224]
        - Wrist camera image: [B, 3, 224, 224]
        - Robot state:        [B, 13]
        - Text instruction:   Tokenized using OpenCLIP

    Output:
        - Normalized 6D action vector:
          [x, y, z, roll, yaw, gripper] in range [-1, 1]
    """

    def __init__(self, state_dim=13, action_dim=ACTION_DIM, device=None):
        super(NiryoVLA, self).__init__()

        # --------------------------------------------------
        # Device Setup
        # --------------------------------------------------
        self.device = device if device is not None else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --------------------------------------------------
        # Vision Encoders (Pretrained ResNet18)
        # --------------------------------------------------
        self.top_encoder = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.top_encoder.fc = nn.Linear(512, 256)

        self.wrist_encoder = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.wrist_encoder.fc = nn.Linear(512, 256)

        # --------------------------------------------------
        # Robot State Encoder (13 -> 64)
        # --------------------------------------------------
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # --------------------------------------------------
        # Text Encoder (OpenCLIP)
        # --------------------------------------------------
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()

        # Tokenizer for text instructions
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Projection from CLIP embedding (512 → 128)
        self.text_projection = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU()
        )

        # --------------------------------------------------
        # Fusion Network
        # 256 (top) + 256 (wrist) + 64 (state) + 128 (text) = 704
        # --------------------------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(704, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Outputs normalized actions in [-1, 1]
        )

        # Move entire model to device
        self.to(self.device)

    # --------------------------------------------------
    # TEXT ENCODING HELPER
    # --------------------------------------------------
    def encode_text(self, instruction, batch_size):
        """
        Encodes a text instruction into a feature vector.
        If instruction is None, returns a zero embedding.
        """
        if instruction is None:
            return torch.zeros(batch_size, 512, device=self.device)

        # Tokenize and move to device
        text_tokens = self.tokenizer([instruction]).to(self.device)

        with torch.no_grad():
            text_feat = self.clip_model.encode_text(text_tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        return text_feat

    # --------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------
    def forward(self, top_img, wrist_img, state, instruction=None, text_tokens=None):
        """
        Forward pass of the VLA model.

        Args:
            top_img: Tensor [B, 3, 224, 224]
            wrist_img: Tensor [B, 3, 224, 224]
            state: Tensor [B, 13]
            instruction: Optional string instruction
            text_tokens: Optional pre-tokenized text

        Returns:
            action: Tensor [B, ACTION_DIM] in normalized range [-1, 1]
        """

        # Ensure inputs are on the correct device
        top_img = top_img.to(self.device)
        wrist_img = wrist_img.to(self.device)
        state = state.to(self.device)

        batch_size = top_img.size(0)

        # --------------------------------------------------
        # Vision Features
        # --------------------------------------------------
        top_feat = self.top_encoder(top_img)
        wrist_feat = self.wrist_encoder(wrist_img)

        # --------------------------------------------------
        # State Features
        # --------------------------------------------------
        state_feat = self.state_encoder(state)

        # --------------------------------------------------
        # Text Features
        # --------------------------------------------------
        if text_tokens is not None:
            text_tokens = text_tokens.to(self.device)
            with torch.no_grad():
                text_feat = self.clip_model.encode_text(text_tokens)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        else:
            text_feat = self.encode_text(instruction, batch_size)

        text_feat = self.text_projection(text_feat)

        # --------------------------------------------------
        # Fusion and Action Prediction
        # --------------------------------------------------
        fused = torch.cat(
            [top_feat, wrist_feat, state_feat, text_feat],
            dim=1
        )

        action = self.fusion(fused)
        return action