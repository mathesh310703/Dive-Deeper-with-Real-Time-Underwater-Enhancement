import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import torchmetrics
import pytorch_ssim  # pip install pytorch-ssim

# -----------------------------
# IMPORT YOUR MODULES
# -----------------------------
from encoder import Encoder
from decoder import Decoder
from loader import CustomLoader
from loss import TemporalLoss

# -----------------------------
# CONFIG
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4  # Reduce if GPU OOM
num_epochs = 20
learning_rate = 1e-4
dataset_root = r"E:\visionenhancementinwarscenarios\data"
checkpoint_dir = os.path.join(dataset_root, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# -----------------------------
# DATASET & DATALOADER
# -----------------------------
train_dataset = CustomLoader(
    dataset_root=dataset_root,
    transform=True,
    domain='underwater'  # Only underwater images
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

print(f"Loaded {len(train_dataset)} underwater samples.")

# -----------------------------
# MODEL
# -----------------------------
encoder = Encoder().to(device)
decoder = Decoder().to(device)

# -----------------------------
# LOSS
# -----------------------------
l1_loss = nn.L1Loss()
temporal_loss = TemporalLoss().to(device)

# -----------------------------
# OPTIMIZER & SCALER
# -----------------------------
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
scaler = GradScaler()

# -----------------------------
# METRICS
# -----------------------------
psnr_metric = torchmetrics.PeakSignalNoiseRatio().to(device)
ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

# -----------------------------
# TRAINING LOOP
# -----------------------------
best_ssim = 0.0

for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    running_loss = 0.0
    pbar = tqdm(train_loader)

    for batch in pbar:
        distorted = batch['distorted'].to(device)
        restored = batch['restored'].to(device)

        optimizer.zero_grad()
        with autocast():
            features = encoder(distorted)
            outputs = decoder(features)

            loss_l1 = l1_loss(outputs, restored)
            loss_ssim = 1 - pytorch_ssim.ssim(outputs, restored)
            loss_temporal = temporal_loss(outputs, restored)
            loss = loss_l1 + 0.2 * loss_ssim + 0.1 * loss_temporal

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_description(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.4f}")

    # -----------------------------
    # METRICS
    # -----------------------------
    encoder.eval()
    decoder.eval()
    psnr_metric.reset()
    ssim_metric.reset()

    with torch.no_grad():
        for batch in train_loader:
            distorted = batch['distorted'].to(device)
            restored = batch['restored'].to(device)
            features = encoder(distorted)
            outputs = decoder(features)
            psnr_metric.update(outputs, restored)
            ssim_metric.update(outputs, restored)

    epoch_psnr = psnr_metric.compute().item()
    epoch_ssim = ssim_metric.compute().item()
    print(f"Epoch [{epoch+1}/{num_epochs}] Avg Loss: {running_loss/len(train_loader):.4f} PSNR: {epoch_psnr:.2f} SSIM: {epoch_ssim:.4f}")

    # -----------------------------
    # SAVE BEST MODEL
    # -----------------------------
    if epoch_ssim > best_ssim:
        best_ssim = epoch_ssim
        torch.save(encoder.state_dict(), os.path.join(checkpoint_dir, "best_encoder.pth"))
        torch.save(decoder.state_dict(), os.path.join(checkpoint_dir, "best_decoder.pth"))
        print(f"Saved best model with SSIM: {best_ssim:.4f}")

print("Training complete!")
