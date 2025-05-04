import streamlit as st
import torch
from torch import nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Define the Generator class
class Generator(nn.Module):
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, noise):
        return self.gen(noise)

# Function to display tensor images
def show_images(images, num_images=25):
    images = (images + 1) / 2  # Rescale to [0,1]
    image_grid = make_grid(images[:num_images], nrow=5).permute(1, 2, 0).detach().cpu()
    fig, ax = plt.subplots()
    ax.imshow(image_grid.squeeze(), cmap="gray")
    ax.axis("off")
    st.pyplot(fig)

# Streamlit UI
st.title("MNIST Digit Generator with DCGAN")

# Load the generator model
z_dim = 64
gen = Generator(z_dim=z_dim)
gen.load_state_dict(torch.load("generator.pth", map_location="cpu"))
gen.eval()

# Input from user
num_samples = st.slider("Number of Digits to Generate", 1, 25, 9)

if st.button("Generate Digits"):
    with torch.no_grad():
        fake_noise = torch.randn(num_samples, z_dim, 1, 1)
        fake_images = gen(fake_noise)
        show_images(fake_images, num_images=num_samples)
