import jax 
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import argparse
from functools import partial

parser = argparse.ArgumentParser(description='Run JAX computation with sharding')
parser.add_argument('--ffn1-only', action='store_true',
                      help='Use only the first FFN layer (skip second matmul and ReLU)')
parser.add_argument('--num-layers', type=int, default=4,
                      help='Number of layers to use')
args = parser.parse_args()
ffn1_only = args.ffn1_only
num_layers = args.num_layers

jax.distributed.initialize()

print("NUM DEVICES: ", jax.devices())
m, n = 16384, 8192
num_devices = 8  # Must match available devices

devices = jax.devices()[:num_devices]
mesh = Mesh(devices, ('i',))  # Bind axis name 'i'

with mesh:
    # Create weights for each layer
    weights = []
    for _ in range(num_layers):
        # First weight matrix for each layer
        x1 = jax.random.normal(jax.random.PRNGKey(0), (n, n), dtype=jnp.bfloat16)
        x1_sharded = jax.device_put(x1, NamedSharding(mesh, P('i', None)))  # Shard cols
        weights.append(x1_sharded)
        
        # Second weight matrix for each layer
        x2 = jax.random.normal(jax.random.PRNGKey(0), (n, n), dtype=jnp.bfloat16)
        x2_sharded = jax.device_put(x2, NamedSharding(mesh, P(None, 'i')))  # Shard rows
        weights.append(x2_sharded)

    # Input tensor
    a = jax.random.uniform(jax.random.PRNGKey(1), (num_devices, m, n), dtype=jnp.bfloat16)
    a_sharded = jax.device_put(a, NamedSharding(mesh, P('i', None, None)))  # Shard batch

@jax.jit
def train_step(weights, a_sharded):
    out, f_vjp = jax.vjp(compute, weights, a_sharded)
    grads = f_vjp(out)
    
    # Apply sharding constraints to enable reduce-scatter
    weight_grads = []
    for i, grad in enumerate(grads[:-1]):  # Last grad is input grad
        if i % 2 == 0:  # First weight matrix in each layer
            weight_grads.append(jax.lax.with_sharding_constraint(grad, NamedSharding(mesh, P('i', None))))
        else:  # Second weight matrix in each layer
            weight_grads.append(jax.lax.with_sharding_constraint(grad, NamedSharding(mesh, P(None, 'i'))))
    
    return out, weight_grads, grads[-1]  # Return output, weight grads, and input grad

if ffn1_only:
    def compute(weights, a_shard):
        out = a_shard
        for i in range(0, len(weights), 2):
            out = jnp.dot(out, weights[i])
        return out
else:
    def compute(weights, a_shard):
        out = a_shard
        for i in range(0, len(weights), 2):
            out = jnp.dot(out, weights[i])
            out = jax.nn.gelu(out)
            out = jnp.dot(out, weights[i+1])
        return out

# Execution
import ctypes
libcudart = ctypes.cdll.LoadLibrary('libcudart.so')

for i in range(50):
    result = train_step(weights, a_sharded)[0].block_until_ready()
    if i == 9:
        libcudart.cudaProfilerStart()
    if i == 19:
        libcudart.cudaProfilerStop()
