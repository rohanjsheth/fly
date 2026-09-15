# Neural Random Features (NRF) from CIFAR-10 via the fly connectome.
# Amid et al., arXiv:2202.06438: each of n_inits random re-initializations emits
# one scalar per image (k=1, so features stay uncorrelated), stacked into
# phi_n(x) = (1/sqrt(n)) [f_1(x), ..., f_n(x)]. So n_inits is the feature
# dimensionality. The connectome fixes the sparsity pattern; only the weight
# values are resampled per init.
#
#   python produce.py --out runs/nrf --n-inits 4     # quick timing check
#   python produce.py --out runs/nrf

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.assign_weights import weight_scale
from utils.make_matrix import gen_matrix
from utils.map_inputs import image_to_activations, map_inputs
from utils.neuron_mapping import map_neurons


def prepare(data_dir):
    neurons = pd.read_parquet(
        data_dir / "neurons.parquet",
        columns=["bodyId", "type", "instance", "assignedOlHex1", "assignedOlHex2"],
    )
    edges = pd.read_parquet(data_dir / "edges.parquet")
    mapped = map_neurons(neurons, edges)
    lookup = map_inputs(mapped["viz_neurons"])

    # Build the CSR once with the He sigmas as its data. The pattern is identical
    # across inits, so per init we only rescale W.data by fresh standard normals.
    mapped["edges"]["neural_weight"] = weight_scale(mapped["neurons"], mapped["edges"])
    return gen_matrix(mapped["edges"], len(mapped["neurons"])), lookup


def load_cifar(root, n_train, n_test, seed, url=None):
    from torchvision import datasets

    if url:  # cs.toronto.edu is slow to the point of stalling; MD5 is still checked
        datasets.CIFAR10.url = url

    rng = np.random.default_rng(seed)
    images, labels, is_test = [], [], []
    for train, count in ((True, n_train), (False, n_test)):
        if count <= 0:
            continue
        ds = datasets.CIFAR10(root=root, train=train, download=True)
        pick = rng.permutation(len(ds.targets))[:count]
        images.append(ds.data[pick])  # (n, 32, 32, 3) uint8, as image_to_activations wants
        labels.append(np.asarray(ds.targets)[pick])
        is_test.append(np.full(len(pick), not train))
    return (np.concatenate(images), np.concatenate(labels).astype(np.int64),
            np.concatenate(is_test))


def make_features(W, viz_idx, drive, steps, batch_size):
    dev = drive.device
    n = W.shape[0]
    n_images = drive.shape[1]
    # Indices are uploaded once and never touched again.
    indptr = torch.as_tensor(W.indptr.astype(np.int32), device=dev)
    indices = torch.as_tensor(W.indices.astype(np.int32), device=dev)
    sigma = torch.as_tensor(W.data.astype(np.float32), device=dev)
    viz = torch.tensor(viz_idx, device=dev)  # copy: to_numpy() is read-only

    def features(seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        values = torch.randn(sigma.shape, generator=g, device=dev) * sigma
        weights = torch.sparse_csr_tensor(indptr, indices, values, size=(n, n))
        # Random linear head, standing in for the paper's final layer.
        head = torch.randn(n, generator=g, device=dev) / math.sqrt(n)

        out = torch.empty(n_images, device=dev)
        for lo in range(0, n_images, batch_size):
            sl = slice(lo, min(lo + batch_size, n_images))
            act = torch.zeros(n, sl.stop - sl.start, device=dev)
            act[viz] = drive[:, sl]
            for _ in range(steps):
                act = torch.relu(weights @ act)
                act[viz] = drive[:, sl]  # inputs are clamped, not evolved
            out[sl] = head @ act
        return out.cpu().numpy()

    return features


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    W, lookup = prepare(Path(args.data))
    images, labels, is_test = load_cifar(args.cifar_root, args.n_train, args.n_test,
                                         args.seed, args.cifar_url)
    np.save(out / "labels.npy", labels)
    np.save(out / "is_test.npy", is_test)

    # Input drive depends only on the image, so it is computed once up front and
    # never re-enters the per-init loop. Keep just the input rows, not all 54,952.
    viz_idx = lookup.index.to_numpy()
    drive = np.stack([image_to_activations(im, lookup, W.shape[0])[viz_idx]
                      for im in images], axis=1)
    drive = torch.as_tensor(drive, device=torch.device(args.device))
    features = make_features(W, viz_idx, drive, args.steps, args.batch_size)

    flops = 2 * W.nnz * len(images) * args.steps * args.n_inits
    print(f"{W.shape[0]:,} neurons, {W.nnz:,} edges, {len(lookup):,} input cells")
    print(f"{len(images):,} images x {args.n_inits:,} inits x {args.steps} steps "
          f"= {flops:.2e} FLOPs on {args.device}")

    seeds = np.random.SeedSequence(args.seed).generate_state(args.n_inits)
    phi = np.zeros((len(images), args.n_inits), dtype=np.float32)
    start = time.perf_counter()
    for i, seed in enumerate(seeds):
        phi[:, i] = features(int(seed))
        if (i + 1) % args.report_every == 0 or i + 1 == args.n_inits:
            done = time.perf_counter() - start
            # Partial saves, so a dropped Colab session still leaves usable columns.
            np.save(out / "features.npy", phi / math.sqrt(args.n_inits))
            print(f"{i + 1}/{args.n_inits} inits  {done / 60:.1f} min elapsed  "
                  f"{flops / args.n_inits * (i + 1) / done / 1e9:.0f} GFLOP/s  "
                  f"eta {done / (i + 1) * (args.n_inits - i - 1) / 60:.1f} min",
                  flush=True)
    print(f"features.npy {phi.shape}  {phi.nbytes / 1e6:.0f} MB")


def parse_args():
    p = argparse.ArgumentParser(description="NRF from CIFAR-10 via the connectome")
    p.add_argument("--out", default="runs/nrf")
    p.add_argument("--data", default="data")
    p.add_argument("--cifar-root", default="data/cifar")
    p.add_argument("--cifar-url", help="mirror for the CIFAR-10 tarball",
                   default="https://cseweb.ucsd.edu/~weijian/static/datasets/cifar/"
                           "cifar-10-python.tar.gz")
    p.add_argument("--n-train", type=int, default=3072)
    p.add_argument("--n-test", type=int, default=0)
    p.add_argument("--n-inits", type=int, default=3072, help="NRF dimensionality")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--report-every", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
