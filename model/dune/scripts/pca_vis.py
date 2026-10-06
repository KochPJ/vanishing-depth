import torch
from pathlib import Path
from PIL import Image
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from model.dune import load_dune_encoder_from_checkpoint
from data.transform import get_test_transform
from data.utils import normalize_min_max


architecture = "vitsmall14"
image_size = 448
checkpoint_path = Path("dune_{}_{}_paper.pth".format(architecture, image_size))
device = torch.device("cpu")
print("Loading DUNE model from checkpoint:", checkpoint_path)
model = load_dune_encoder_from_checkpoint(checkpoint_path)[0]
model = model.eval()
if torch.cuda.is_available():
    print(" - Using GPU")
    device = torch.device("cuda")
model = model.to(device)

print("Loading the test image")
transform = get_test_transform(image_size)
image = Image.open("./assets/test_image.png").convert("RGB")
image = transform(image)

print("Making a forward pass through the model")
with torch.inference_mode():
    output: dict = model(image.unsqueeze(0).to(device))
    # output is compatibile to that of DINOv2
    # output["x_norm_clstoken"].shape --> [1, 768]
    # output["x_norm_patchtokens"].shape --> [1, num_patches, 768]
    patch_emb = output["x_norm_patchtokens"].detach().cpu().squeeze()

print("Reducing the dimension of patch embeddings to 3 via PCA")
num_patches_side = int(patch_emb.shape[0] ** 0.5)  # assume a square image
pca = PCA(n_components=3, random_state=22)
patch_pca = torch.from_numpy(pca.fit_transform(patch_emb.numpy()))  # [num_patches, 3]
patch_pca = patch_pca.reshape([num_patches_side, num_patches_side, 3]).permute(2, 0, 1)
patch_pca = (
    torch.nn.functional.interpolate(patch_pca.unsqueeze(0), image_size, mode="nearest")
    .squeeze(0)
    .permute(1, 2, 0)
)  # [image_size, image_size, 3]
patch_pca = normalize_min_max(patch_pca)

print("Visualizing the original image and the PCA-reduced patch embeddings")
plt.close()
fig, axs = plt.subplots(1, 2, dpi=200, constrained_layout=True)
axs[0].imshow(normalize_min_max(image.permute(1, 2, 0)))
axs[1].imshow(patch_pca)
for ax in axs:
    ax.axis("off")
plt.savefig(
    "./assets/test_image_patch_pca_{}.png".format(checkpoint_path.stem),
    bbox_inches="tight",
)
