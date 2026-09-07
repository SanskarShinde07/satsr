import tacoreader.v1 as tacoreader
import rasterio as rio
import matplotlib.pyplot as plt

dataset = tacoreader.load("tacofoundation:sen2naipv2-unet")
print("Dataset loaded:", len(dataset), "samples")

sample_idx = 4000
lr = dataset.read(sample_idx).read(0)
hr = dataset.read(sample_idx).read(1)

with rio.open(lr) as src, rio.open(hr) as dst:
    lr_data = src.read()
    hr_data = dst.read()

print("LR shape:", lr_data.shape)
print("HR shape:", hr_data.shape)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(lr_data[:3].transpose(1, 2, 0) / 3000)
ax[0].set_title("LR (10 m)")
ax[0].axis("off")
ax[1].imshow(hr_data[:3].transpose(1, 2, 0) / 3000)
ax[1].set_title("HR (2.5 m)")
ax[1].axis("off")
plt.tight_layout()
plt.savefig("outputs/sample_check.png", dpi=120)
print("Saved to outputs/sample_check.png")