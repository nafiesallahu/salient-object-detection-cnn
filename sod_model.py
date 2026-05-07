import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_TYPES = ("baseline", "baseline_no_bn", "unet_small")


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineSODCNN(nn.Module):
    """Simple scratch-built encoder-decoder CNN for saliency masks."""

    def __init__(self, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.enc1 = self._encoder_layer(3, 32, use_batchnorm)
        self.enc2 = self._encoder_layer(32, 64, use_batchnorm)
        self.enc3 = self._encoder_layer(64, 128, use_batchnorm)
        self.enc4 = self._encoder_layer(128, 256, use_batchnorm)

        self.up4 = self._decoder_layer(256, 128, use_batchnorm)
        self.up3 = self._decoder_layer(128, 64, use_batchnorm)
        self.up2 = self._decoder_layer(64, 32, use_batchnorm)
        self.up1 = self._decoder_layer(32, 16, use_batchnorm)

        self.output_conv = nn.Conv2d(16, 1, kernel_size=1)

    @staticmethod
    def _encoder_layer(
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool,
    ) -> nn.Sequential:
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=not use_batchnorm,
            ),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.extend([nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)])
        return nn.Sequential(*layers)

    @staticmethod
    def _decoder_layer(
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool,
    ) -> nn.Sequential:
        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
            ),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        x = self.up4(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)

        logits = self.output_conv(x)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(logits)


class BaselineSODCNNNoBatchNorm(BaselineSODCNN):
    """Same baseline architecture with BatchNorm layers removed for ablation."""

    def __init__(self) -> None:
        super().__init__(use_batchnorm=False)


class SmallUNet(nn.Module):
    """Small UNet-style encoder-decoder with skip connections."""

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in the range [0, 1).")

        self.enc1 = ConvBlock(3, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256, dropout=dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(256, 512, dropout=dropout)

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512, 256, dropout=dropout)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        self.output_conv = nn.Conv2d(32, 1, kernel_size=1)

    @staticmethod
    def _match_size(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != reference.shape[-2:]:
            x = F.interpolate(
                x,
                size=reference.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        bottleneck = self.bottleneck(self.pool(e4))

        d4 = self._match_size(self.up4(bottleneck), e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self._match_size(self.up3(d4), e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self._match_size(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self._match_size(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        logits = self.output_conv(d1)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(logits)


def get_model(model_type: str = "baseline", dropout: float = 0.1) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "baseline":
        return BaselineSODCNN()
    if model_type == "baseline_no_bn":
        return BaselineSODCNNNoBatchNorm()
    if model_type == "unet_small":
        return SmallUNet(dropout=dropout)
    raise ValueError(f"model_type must be one of: {', '.join(MODEL_TYPES)}")
