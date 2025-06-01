import jax.numpy as jnp
import numpy as np
from torch.utils.data import default_collate
from einops import rearrange
from PIL import Image
import jax


def count_params(param_dict):
    leaves = jax.tree_util.tree_leaves(param_dict)
    return sum([jnp.size(p) for p in leaves])


def pil_to_jax(img, ismnist=False):
    """Convert a PIL Image (H, W, C) to (C, H, W) float32 jax.numpy array scaled to [-1, 1]."""
    arr = jnp.asarray(img, dtype=jnp.float32) / 255.0
    # arr = (arr - 0.5) / 0.5  # normalize to [-1, 1]
    if not ismnist:
        arr = rearrange(arr, "h w c -> c h w")
    return arr


def jax_to_pil(img_jnp):
    """
    Convert a (C, H, W) float32 JAX array in the range [-1, 1]—as produced by
    `ToArrayJNP`—back into a standard Pillow Image.

    Args:
        img_jnp (jax.numpy.ndarray): Image tensor (C, H, W) with values in [-1, 1].

    Returns:
        PIL.Image.Image: Reconstructed image in RGB (or grayscale) format.
    """
    # De-normalize from [-1, 1] → [0, 255]
    img_jnp = (img_jnp * 0.5 + 0.5) * 255.0
    img_jnp = jnp.clip(img_jnp, 0, 255).astype(jnp.uint8)
    img_jnp = rearrange(img_jnp, "b h w -> h w b")
    # Drop the channel dim if single‑channel (grayscale)
    if img_jnp.shape[2] == 1:
        img_jnp = img_jnp[:, :, 0]
    return Image.fromarray(np.array(img_jnp))


# taken from https://cloud.google.com/blog/products/ai-machine-learning/guide-to-jax-for-pytorch-developers
def jax_collate(batch):
    return jax.tree.map(jnp.asarray, default_collate(batch))
