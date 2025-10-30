import flax.nnx as nnx
import jax
import jax.numpy as jnp


class ScaledDotProductAttentionLayer(nnx.Module):
    def __init__(self, rngs: nnx.Rngs) -> None:
        self.WK = jnp.zeros(12)
        self.WQ = jnp.zeros(12)
        self.WV = jnp.zeros(12)

        self.layer_norm = nnx.LayerNorm(12, rngs=rngs)

        self.mlp1 = nnx.Linear(in_features=..., out_features=..., rngs=rngs)
        self.mlp2 = nnx.Linear(in_features=..., out_features=..., rngs=rngs)

        x_dim = 12
        self.D = jnp.float32(x_dim)

    def __call__(self, X: jax.Array) -> jax.Array:
        """Calculate multi-head attention

        H =
        """
        K_BD = X @ self.WK
        Q_BD = X @ self.WQ
        V_BD = X @ self.WV

        A_BB = nnx.softmax(Q_BD @ K_BD.T / self.D, axis=1)
        H_BD = A_BB @ V_BD

        # jnp.einsum()
        jnp.einsum("hij,jk,kl->il")

        Z_BD = self.layer_norm(H_BD)

        Z_BD = nnx.relu(self.mlp1(Z_BD))
        Z_BD = nnx.relu(self.mlp2(Z_BD))


class Transformer(nnx.Module):
    def __init__(self): ...

    def __call__(self, x: jax.Array): ...


# step 1 - word embedding (taking integer indices for each char)
# step 2 - positional encoding
# step 3 - transformer layers (multi-head attention -> LayerNorm -> MLP -> LayerNorm)
#          - try to figure out the one-liner with einsum
# step 4 - loss func
# step 5 - idk think that's it?
