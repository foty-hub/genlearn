import jax.numpy as jnp
import flax.linen as nn

from einops import rearrange
from distrax import MultivariateNormalDiag


class VAE(nn.Module):
    enc_hidden_layers: list[int]
    dec_hidden_layers: list[int]

    latent_dim: int = 2
    dec_out: int = 784  # 28x28
    out_width: int = 28

    @nn.compact
    def __call__(self, x):
        q_mean, q_logvar = self.encode(x)
        z = self.sample_latent(q_mean, q_logvar)
        x_hat = self.decode(z)
        return x_hat, (q_mean, q_logvar)

    @nn.compact
    def encode(self, x):
        x = rearrange(x, "b h w -> b (h w)")  # flatten
        # forward pass
        for i, n_features in enumerate(self.enc_hidden_layers):
            x = nn.Dense(n_features, use_bias=False, name=f"encoder_dense{i}")(x)
            x = nn.relu(x)
        # return two outputs - the mean and log variance of the latent distribution, with an entry per latent dimension
        mean_bz = nn.Dense(self.latent_dim, name="encoder_mean_out")(x)
        logvar_bz = nn.Dense(self.latent_dim, name="encoder_logvar_out")(x)
        return mean_bz, logvar_bz

    @nn.compact
    def sample_latent(self, mean_bz, logvar_bz):
        std_bz = jnp.exp(0.5 * logvar_bz)
        dist = MultivariateNormalDiag(mean_bz, std_bz)
        return dist.sample(seed=self.make_rng("params"))

    @nn.compact
    def decode(self, z):
        # forward pass
        for i, n_features in enumerate(self.dec_hidden_layers):
            z = nn.Dense(n_features, use_bias=False, name=f"decoder_dense{i}")(z)
            z = nn.relu(z)
        z = nn.Dense(self.dec_out, name="decoder_out")(z)
        z = nn.sigmoid(z)
        z = rearrange(z, "b (w h) -> b w h", w=self.out_width)  # unflatten
        return z


if __name__ == "__main__":
    # example usage
    ...
