import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class TemporalLoss(nn.Module):
    def __init__(self):
        super(TemporalLoss, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.l2 = nn.MSELoss()
        self.previous_vgg_features = None

    def forward(self, outputs, target):
        # Extract VGG features
        current_vgg_features = self.vgg(outputs)
        target_vgg_features = self.vgg(target)

        # Content loss
        loss_content = self.l2(current_vgg_features, target_vgg_features)

        # Temporal loss
        if self.previous_vgg_features is None or self.previous_vgg_features.size(0) != current_vgg_features.size(0):
            temporal_loss = torch.tensor(0.0, device=outputs.device)
        else:
            temporal_loss = self.l2(self.previous_vgg_features, current_vgg_features)

        self.previous_vgg_features = current_vgg_features.detach()

        return loss_content + temporal_loss
