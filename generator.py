import torch
import torch.nn as nn
import torchvision.utils as vutils


latent_dim = 100
img_size = 28
img_shape = (1, img_size, img_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


generator = Generator().to(device)
generator.load_state_dict(torch.load('G&D\generator.pth.'))
generator.eval()


z = torch.randn(64, latent_dim, device=device)
generated_imgs = generator(z)


vutils.save_image(generated_imgs.data, 'Image/generated_samples.png', nrow=8, normalize=True)
