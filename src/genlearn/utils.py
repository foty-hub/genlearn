import jax.numpy as jnp
import numpy as np
from einops import rearrange
from PIL import Image
import jax

__all__ = ["count_params", "pil_to_jax", "jax_to_pil", "jax_collate"]


def count_params(param_dict):
    """Count total number of parameters in a JAX parameter dict."""
    leaves = jax.tree_util.tree_leaves(param_dict)
    return sum([jnp.size(p) for p in leaves])


def pil_to_jax(img, ismnist=False):
    """Convert a PIL Image to a JAX array scaled to [0,1] or rearranged for non-MNIST images."""
    arr = jnp.asarray(img, dtype=jnp.float32) / 255.0
    if not ismnist:
        arr = rearrange(arr, "h w c -> c h w")
    return arr


def jax_to_pil(img_jnp):
    """Convert a JAX array (C, H, W) in [-1,1] back to a PIL Image."""
    img = (img_jnp * 0.5 + 0.5) * 255.0
    img = jnp.clip(img, 0, 255).astype(jnp.uint8)
    img = rearrange(img, "c h w -> h w c")
    np_img = np.array(img)
    if np_img.shape[2] == 1:
        np_img = np_img[:, :, 0]
    return Image.fromarray(np_img)


def jax_collate(batch):
    """Collate function for JAX that stacks images and labels."""
    imgs, labels = zip(*batch)
    return jnp.stack(imgs), jnp.array(labels)
