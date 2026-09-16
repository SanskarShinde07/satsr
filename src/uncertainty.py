import numpy as np
import torch


def uncertainty_map(model, x, device="cpu"):
    """
    Run the model under 8 flips/rotations, undo each transform,
    and measure disagreement per pixel.

    x: tensor (1, C, H, W), values 0-1
    returns: (mean_sr, uncertainty) as numpy arrays
    """
    outs = []

    for k in range(2):                      # 4 rotations
        for flip in (False,):         
            xi = torch.rot90(x, k, dims=(2, 3))
            if flip:
                xi = torch.flip(xi, dims=(3,))

            with torch.no_grad():
                yi = model(xi.to(device)).clamp(0, 1).cpu()

            if flip:
                yi = torch.flip(yi, dims=(3,))
            yi = torch.rot90(yi, -k, dims=(2, 3))

            outs.append(yi)

    stack = torch.stack(outs)               # (8, 1, C, H, W)
    mean = stack.mean(0).squeeze(0).numpy()
    std  = stack.std(0).squeeze(0).numpy()

    # average disagreement across channels -> single map
    unc = std.mean(axis=0)
    return mean, unc