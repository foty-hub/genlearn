import pathlib

import jax
import jax.numpy as jnp
import numpy as np

PAD = 0
START = 1
END = 2
MAXLEN = 128


def _encode_char(char: str) -> int:
    "The range of ords in shakespeare is 32-122. We subtract 27, compressing that range to 5-95. The first values are left empty for special tokens (start, stop, pad?)"
    return ord(char) - 27


def _decode_char(token: jnp.integer) -> str:
    return chr(token + 27)


def encode(string: str, length: int = MAXLEN) -> jax.Array:
    chars = [_encode_char(c) for c in string]
    arr = jnp.array([START, *chars, END], dtype=jnp.uint8)
    return jnp.pad(arr, pad_width=(0, length - len(arr)), mode="constant")


def decode(tokens: jax.Array) -> str:
    return "".join(_decode_char(tok) for tok in tokens[1:-1])


def get_shakespeare_dataset() -> jax.Array:
    """Load the Shakespeare dataset as a jax array, with the following token values:

    0 - pad
    1 - start
    2 - end
    rest = ord(c) - 27

    Padded to length 128
    """
    data_path = pathlib.Path(__file__).parents[3] / "data" / "shakespeare.txt"

    with open(data_path, "r") as f:
        data = f.readlines()

    data = [d.strip() for d in data if d != "\n"]  # remove the newlines

    dataset = jnp.array([encode(s) for s in data])
    return dataset


def main():
    dataset = get_shakespeare_dataset()
    print(f"{dataset.shape=}")


if __name__ == "__main__":
    main()
