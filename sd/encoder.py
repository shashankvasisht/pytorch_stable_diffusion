import torch 
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBllock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            # (Batch_size, Channel, Height, Width) -> (Batch_size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            # (Batch_size, 128, Height, Width) -> (Batch_size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),

            # (Batch_size, 128, Height, Width) -> (Batch_size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),

            # (Batch_size, 128, Height, Width) -> (Batch_size, 128, Height/2, Width/2)
            nn.Conv2d(128, 128, kernel_size=2, stride=2, padding=0),

            # (Batch_size, 128, Height/2, Width/2) -> (Batch_size, 256, Height/2, Width/2)
            VAE_ResidualBlock(128, 256),

            # (Batch_size, 256, Height/2, Width/2) -> (Batch_size, 256, Height/2, Width/2)
            VAE_ResidualBlock(256, 256),

            # (Batch_size, 256, Height/2, Width/2) -> (Batch_size, 256, Height/4, Width/4)
            nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=0),

            # (Batch_size, 256, Height/4, Width/4) -> (Batch_size, 512, Height/4, Width/4)
            VAE_ResidualBlock(256, 512),

            # (Batch_size, 512, Height/4, Width/4) -> (Batch_size, 512, Height/4, Width/4)
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height/4, Width/4) -> (Batch_size, 512, Height/8, Width/8)
            nn.Conv2d(512, 512, kernel_size=2, stride=2, padding=0),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            VAE_AttentionBllock(512),  # RUNS SELF ATTENTION OVER EACH PIXEL --> Global Context

            # (Batch_size, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            nn.GroupNorm(32,512), # Num groups = 32, num_channels = 512

            # (Batch_size, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            nn.SiLU(), # Sigmoid Linear Unit 

            # (Batch_size, 512, Height/8, Width/8) -> (Batch_size, 8, Height/8, Width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (Batch_size, 8, Height/8, Width/8) -> (Batch_size, 8, Height/8, Width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)

        )

    def forward(self, x: torch.Tensor, noise : torch.Tensor) -> torch.Tensor:
        # x : (Batch_Size, Channel, Height, Width)
        # noise : (Batch_Size, Out_Channels, Height/8, Width/8)

        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom)
                x = F.pad(x, (0, 1, 0, 1)) # To the convs with stride 2, apply assymetric padding on right and bottom
                x = module(x)


        # Now it's not just an autoencoder, but it's a variational autoencoder. This means that 
        # the output of the encoder is not a latent image but rather the mean and variance
        # of a multivariate gaussian, which is essentially learnt from the distribution of our data.

        #Thus we will get mean and log_variance from the output as follows. We will use torch.chunk
        # to break the tensor with channel 8 -> to two equal tensors of channel 4 respectively.

        # (Batch_Size, 8, Height/8, Width/8) -> two tensors of shape (Batch_Size, 4, Height/8, Width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # We'll clamp this variance between a certain range 
        # (Batch_Size, 4, Height/8, Width/8) -> (Batch_Size, 4, Height/8, Width/8)
        log_variance = torch.clamp(log_variance, -30, 20)

        # (Batch_Size, 4, Height/8, Width/8) -> (Batch_Size, 4, Height/8, Width/8)
        # To get variance from log variance we need to take it's exponent
        variance = log_variance.exp()

        # (Batch_Size, 4, Height/8, Width/8) -> (Batch_Size, 4, Height/8, Width/8)
        #To get stdev from variance, take it's sqrt
        stdev = variance.sqrt()

        # Now, how do we sample something from this distribution???
        # Remember that we have a noise vector given to us (Let's say Z) => Z=N(0,1)
        # If we want to change this noise to our derived distribution we can add the
        # new distribution's mean and multiply with the stdev of the new dist.

        # Z = N(0,1) -> X = N(mean, variance)
        x = mean + stdev * noise

        # Scale the output by a constant 
        x *= 0.18215

        return x
    



