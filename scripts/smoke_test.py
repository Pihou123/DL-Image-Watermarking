from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models.hidden_system import HiddenSystem
from src.noise.manager import NoiseManager


def main() -> None:
    device = torch.device("cpu")

    model_cfg = {
        "message_length": 30,
        "encoder_blocks": 3,
        "encoder_channels": 64,
        "decoder_blocks": 3,
        "decoder_channels": 64,
        "discriminator_blocks": 3,
        "discriminator_channels": 64,
        "use_discriminator": True,
        "use_vgg": False,
        "loss_weights": {"encoder": 1.0, "decoder": 1.0, "adversarial": 1.0},
    }
    train_cfg = {"lr_encoder_decoder": 1e-3, "lr_discriminator": 1e-3}
    noise_cfg = {"strategy": "single_random", "layers": [{"name": "identity", "probability": 1.0, "params": {}}]}

    noise_manager = NoiseManager(noise_cfg, device=device)
    system = HiddenSystem(model_cfg, train_cfg, image_size=(64, 64), noise_manager=noise_manager, device=device)

    images = torch.randn(2, 3, 64, 64)
    messages = torch.randint(0, 2, (2, 30), dtype=torch.float32)

    encoded, noised, decoded, meta = system.infer(images, messages, epoch=1)
    print("encoded", tuple(encoded.shape))
    print("noised", tuple(noised.shape))
    print("decoded", tuple(decoded.shape))
    print("noise", meta)


if __name__ == "__main__":
    main()

