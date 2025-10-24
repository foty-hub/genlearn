import flax.linen as nn
import jax
import jax.numpy as jnp


class Transformer(nn.Module):
    def __init__(self): ...

    def __call__(self, x: jax.Array): ...
