import torch
import torch.nn as nn
from torchvision import models
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")  # Suppress deprecations

class DSHTEEnsemble(nn.Module):
    """
    Simplified DSHTE: CNN Ensemble with Max Voting.
    3 backbones: MobileNetV3, EfficientNet-B0, ResNet-18.
    Transfer learning: Freeze early layers.
    """
    def __init__(self, num_classes=26):
        super().__init__()
        self.num_classes = num_classes
        
        # Load pre-trained with weights (fixes warnings)
        self.mobilenet = models.mobilenet_v3_small(weights='DEFAULT')
        self.mobilenet.classifier[3] = nn.Linear(self.mobilenet.classifier[3].in_features, num_classes)
        for param in self.mobilenet.features[:6].parameters():  # Freeze early
            param.requires_grad = False
        
        self.efficientnet = models.efficientnet_b0(weights='DEFAULT')
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)
        for param in self.efficientnet.features[:4].parameters():
            param.requires_grad = False
        
        self.resnet = models.resnet18(weights='DEFAULT')
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        for param in self.resnet.layer1.parameters():  # Freeze early
            param.requires_grad = False
    
    def forward(self, x):
        # Get logits from each
        out_mob = self.mobilenet(x)
        out_eff = self.efficientnet(x)
        out_res = self.resnet(x)
        
        # Stack for batch voting later
        return torch.stack([out_mob, out_eff, out_res], dim=1)  # [B, 3, C]
    
    def predict_with_voting(self, x):
        """Max voting: Argmax on stacked probs."""
        with torch.no_grad():
            logits = self(x)  # [B, 3, C]
            probs = torch.softmax(logits, dim=-1)
            # Voting: Mean probs then argmax (soft voting)
            mean_probs = probs.mean(dim=1)
            preds = mean_probs.argmax(dim=1)
            confs = mean_probs.max(dim=1)[0]
        return preds, confs