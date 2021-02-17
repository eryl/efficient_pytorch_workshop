import argparse

import numpy as np
import torch

def main():
    parser = argparse.ArgumentParser(description='Script to experiment with pipelining of data transfer to GPU ')
    parser.add_argument('--pipeline', default=False, action='store_true')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    rng = np.random.default_rng(1729)

    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA not available and device set to {args.device}")
        else:
            device = torch.device(args.device)

    print(f"Device is set to {device}")

    n = 2048
    a = torch.tensor(rng.random((n, n)))
    b = torch.tensor(rng.random((n, n)))
    max_epochs = 10

    means = []
    if args.pipeline:
        current_a = a.to(device)
        current_b = b.to(device)
        # We perform one less iteration in the loop, since we need to explicitly perform it once more after the loop on
        # the last batch transferred to be comparable with the non-pipelined case
        for i in range(max_epochs-1):
            # The idea is that by making the transfer here, before we compute on the batches we moved in the previous
            # iteration, the transfer latency would be hidden. This doesn't seem to work as I expect
            next_a = a.to(device)
            next_b = b.to(device)
            c_d = current_a @ current_b
            c = c_d.cpu()
            means.append(c.mean().item())
            current_a = next_a
            current_b = next_b
        # To perform the same amount of computation, we
        c_d = current_a @ current_b
        c = c_d.cpu()
        means.append(c.mean().item())
    else:
        for i in range(max_epochs):
            a_d = a.to(device)
            b_d = b.to(device)
            c_d = a_d @ b_d
            c = c_d.cpu()
            means.append(c.mean().item())


if __name__ == '__main__':
    main()

