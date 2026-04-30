import torch.nn as nn


class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # ===== FEATURE EXTRACTION =====
        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3 (NEW - improves performance)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # ===== CLASSIFIER =====
        self.classifier = nn.Sequential(
            nn.Flatten(),

            # ⚠️ IMPORTANT: matches 128x128 input after pooling
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),

            # Regularization
            nn.Dropout(0.5),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x