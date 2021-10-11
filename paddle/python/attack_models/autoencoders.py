from paddle import nn

class Autoencoder(nn.Layer):
    def __init__(self, channels=3):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2D(channels, 16, 4, stride=2, padding=1),
            nn.BatchNorm2D(16),
            nn.ReLU(),
            nn.Conv2D(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2D(32),
            nn.ReLU(),
            nn.Conv2D(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2D(128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2DTranspose(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2DTranspose(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2D(32),
            nn.ReLU(),
            nn.Conv2DTranspose(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2D(16),
            nn.ReLU(),
            nn.Conv2DTranspose(16, channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x