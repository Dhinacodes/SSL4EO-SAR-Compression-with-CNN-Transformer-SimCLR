class CNNTransformerEncoder(nn.Module):
    def __init__(self, input_channels=12, compressed_dim=1024):
        super(CNNTransformerEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512), nn.ReLU(),
        )
        self.proj = nn.Linear(512, 128)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=256, batch_first=True),
            num_layers=2
        )
        self.fc = nn.Linear(128, compressed_dim)

    def forward(self, x):
        x = self.cnn(x)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)
        x = self.proj(x)
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))
