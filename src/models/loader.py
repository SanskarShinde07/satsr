import torch
from src.models.rrdbnet import RRDBNet


def load_pretrained(ckpt_path, num_ch=4, device="cpu"):
    model = RRDBNet(num_in_ch=num_ch, num_out_ch=num_ch,
                    num_feat=64, num_block=23, num_grow_ch=32)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("params_ema", ckpt.get("params", ckpt))

    if num_ch != 3:
        # conv_first: [64, 3, 3, 3] -> [64, num_ch, 3, 3]
        w = sd["conv_first.weight"]
        extra = w[:, 0:1].repeat(1, num_ch - 3, 1, 1)   # copy red
        sd["conv_first.weight"] = torch.cat([w, extra], dim=1)

        # conv_last: [3, 64, 3, 3] -> [num_ch, 64, 3, 3]
        w = sd["conv_last.weight"]
        extra = w[0:1].repeat(num_ch - 3, 1, 1, 1)
        sd["conv_last.weight"] = torch.cat([w, extra], dim=0)

        b = sd["conv_last.bias"]
        sd["conv_last.bias"] = torch.cat([b, b[0:1].repeat(num_ch - 3)])

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    return model.to(device).eval()