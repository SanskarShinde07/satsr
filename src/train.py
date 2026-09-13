import os, time, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.dataset import Sen2NaipDataset
from src.models.loader import load_pretrained

CKPT_IN  = "checkpoints/esrgan_1S2.pth"
CKPT_OUT = "/kaggle/working/satsr_finetuned.pth"
EPOCHS, BATCH, LR, N = 6, 8, 1e-4, 4000


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)

    model = load_pretrained(CKPT_IN, num_ch=3, device=device)
    model.train()

    loader = DataLoader(Sen2NaipDataset(N), batch_size=BATCH,
                        shuffle=True, num_workers=0, drop_last=True)
    print("Dataloader ready, starting training", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.L1Loss()

    start = 0
    if os.path.exists(CKPT_OUT):
        ck = torch.load(CKPT_OUT, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["epoch"] + 1
        print("Resumed from epoch", start, flush=True)

    for ep in range(start, EPOCHS):
        t0, total = time.time(), 0.0
        for i, (lr_img, hr_img) in enumerate(loader):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            opt.zero_grad()
            loss = crit(model(lr_img), hr_img)
            loss.backward()
            opt.step()
            total += loss.item()
            if i % 10 == 0:
                print(f"ep {ep} step {i}/{len(loader)} loss {loss.item():.4f}",
                      flush=True)

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep}, CKPT_OUT)
        print(f"== epoch {ep} done  avg {total/len(loader):.4f}  "
              f"{time.time()-t0:.0f}s  saved ==", flush=True)


if __name__ == "__main__":
    main()