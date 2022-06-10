import io
import math
import numpy as np
import os
import torchvision
from data import era5, fastmri, image
from helpers import get_dataset_root
from PIL import Image, ImageOps
from pathlib import Path
from tqdm import tqdm


def SaveWithTargetSize(img, target_in_bytes, format: str):
    if format == "jpeg":
        jpeg_img, num_bytes = JPEGSaveWithTargetSize(img, target_in_bytes)
    elif format == "jpeg2000":
        jpeg_img, num_bytes = JPEG2000SaveWithTargetSize(img, target_in_bytes)
    elif format == "bpg":
        jpeg_img, num_bytes = BPGSaveWithTargetSize(img, target_in_bytes)
    else:
        raise ValueError

    return jpeg_img, num_bytes


def JPEGSaveWithTargetSize(im, target_in_bytes):
    """ """
    Qmin, Qmax = 1, 96  # Quality range for JPEG
    Qacc = -1  # Acceptable quality for target size

    while Qmin <= Qmax:
        q = math.floor((Qmin + Qmax) / 2)

        # Encode into memory and get size
        buffer = io.BytesIO()
        im.save(buffer, format="JPEG", quality=q)
        s = buffer.getbuffer().nbytes

        if s <= target_in_bytes:
            Qacc = q
            Qmin = q + 1
        elif s > target_in_bytes:
            Qmax = q - 1

    # Return PIL image and size in bytes
    if Qacc > -1:
        return Image.open(buffer), buffer.getbuffer().nbytes
    else:
        raise Exception("ERROR: No acceptable quality factor found")


def get_jpeg2000_enc_cmd(in_filepath, quant, out_filepath):
    return f"opj_compress -i {in_filepath} -r {quant} -o {out_filepath} 2>1 1>/dev/null"


def get_jpeg2000_dec_cmd(in_filepath, out_filepath):
    return f"opj_decompress -i {in_filepath} -o {out_filepath} 2>1 1>/dev/null"


def JPEG2000SaveWithTargetSize(im, target_in_bytes):
    """ """
    # Higher quantization means lower quality, so quant_min corresponds to highest
    # quality and quant_max to lowest quality
    quant_min, quant_max = 2, 96
    quant_acc = -1  # Acceptable quality for target size

    # Save original image
    in_filepath = str(Path(os.getcwd()) / "in.png")
    out_filepath = str(Path(os.getcwd()) / "out.jp2")
    out_filepath_png = str(Path(os.getcwd()) / "out.png")
    im.save(in_filepath, format="PNG")

    while quant_max >= quant_min:
        quant = math.floor((quant_max + quant_min) / 2)

        # Encode image and get size
        os.system(get_jpeg2000_enc_cmd(in_filepath, quant, out_filepath))
        s = os.path.getsize(out_filepath)

        if s <= target_in_bytes:
            quant_acc = quant
            quant_max = quant - 1
        elif s > target_in_bytes:
            quant_min = quant + 1

    # Return PIL image and size in bytes for best acceptable quality
    if quant_acc > -1:
        os.system(get_jpeg2000_enc_cmd(in_filepath, quant_acc, out_filepath))
        s = os.path.getsize(out_filepath)
        # Decode final jpeg2000 image to png so we can open it with PIL
        os.system(get_jpeg2000_dec_cmd(out_filepath, out_filepath_png))
        return Image.open(out_filepath_png), s
    else:
        raise Exception("ERROR: No acceptable quality factor found")


def get_bpg_enc_cmd(in_filepath, quant, out_filepath):
    return f"bpgenc -f 444 -q {quant} -o {out_filepath} {in_filepath}"


def get_bpg_dec_cmd(in_filepath, out_filepath):
    return f"bpgdec -o {out_filepath} {in_filepath}"


def BPGSaveWithTargetSize(im, target_in_bytes):
    """ """
    # Higher quantization means lower quality, so quant_min
    # corresponds to highest quality and quant_max to lowest
    # quality
    quant_min, quant_max = 0, 51
    quant_acc = -1  # Acceptable quality for target size

    # Save original image
    in_filepath = str(Path(os.getcwd()) / "in.png")
    out_filepath = str(Path(os.getcwd()) / "out.bpg")
    out_filepath_png = str(Path(os.getcwd()) / "out.png")
    im.save(in_filepath, format="PNG")

    while quant_max >= quant_min:
        quant = math.floor((quant_max + quant_min) / 2)

        # Encode image and get size
        os.system(get_bpg_enc_cmd(in_filepath, quant, out_filepath))
        s = os.path.getsize(out_filepath)

        if s <= target_in_bytes:
            quant_acc = quant
            quant_max = quant - 1
        elif s > target_in_bytes:
            quant_min = quant + 1

    # Return PIL image and size in bytes for best acceptable quality
    if quant_acc > -1:
        os.system(get_bpg_enc_cmd(in_filepath, quant_acc, out_filepath))
        s = os.path.getsize(out_filepath)
        # Decode final bpg image to png so we can open it with PIL
        os.system(get_bpg_dec_cmd(out_filepath, out_filepath_png))
        return Image.open(out_filepath_png), s
    else:
        raise Exception("ERROR: No acceptable quality factor found")


def pil_to_array(img):
    return np.asarray(img).astype(float) / 255.0


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    if mse == 0:
        raise ValueError
    return -10.0 * np.log10(mse)


def run(dataset_name, bpp, format):
    # Load dataset
    if dataset_name == "era5":
        ds = era5.ERA5(root=get_dataset_root("era5"), split="test")
    elif dataset_name == "cifar10":
        ds = torchvision.datasets.CIFAR10(root=get_dataset_root("cifar10"), train=False)
    elif dataset_name == "fastmri":
        ds = fastmri.FastMRI(
            root=get_dataset_root("fastmri"),
            split="val",
            challenge="multicoil",
        )
    elif dataset_name == "kodak":
        ds = image.Kodak(
            root=get_dataset_root("kodak"),
            download=True,
        )

    # Convert bpp targets to byte targets
    if dataset_name == "cifar10":
        target_in_bytes = bpp * 32 * 32 / 8  # 32 * 32 pixels, 8 bits per byte
    elif dataset_name == "era5":
        target_in_bytes = bpp * 46 * 90 / 8  # 46 * 90 temperatures, 8 bits per byte
    elif dataset_name == "fastmri":
        target_in_bytes = None  # Each MRI scan has a different shape, so set target_in_bytes per instance
    elif dataset_name == "kodak":
        target_in_bytes = bpp * 768 * 512 / 8

    # Iterate over dataset and compute best image that can be saved with less
    # than target_in_bytes bytes
    psnr_vals = []
    for i in tqdm(range(len(ds))):
        if dataset_name == "fastmri":
            # Remove channel dimension to get array of shape (depth, height, width)
            mri = ds[i][0].numpy()
            depth, height, width = mri.shape
            # The target image size for each slice is given by following formula, since
            # slices have shape height * width and there are 8 bits per byte
            target_in_bytes = bpp * height * width / 8
            # Initialize MRI reconstruction (where each slice will be filled out)
            mri_reconstruction = np.zeros(mri.shape)
            # Iterate over slices and compress them individually
            for d in range(depth):
                mri_slice = mri[d]  # Shape (height, width)
                # Convert MRI slice to grayscale "image"
                img = Image.fromarray(np.uint8(mri_slice * 255.0), "L")
                # Compress slice
                jpeg_img, num_bytes = SaveWithTargetSize(img, target_in_bytes, format)

                if dataset_name in ["fastmri", "era5"]:  # 💩 hack for greyscale
                    jpeg_img = ImageOps.grayscale(jpeg_img)

                # Add compressed slice to total reconstruction (for other compression
                # modalities we might need to be careful with shapes)
                mri_reconstruction[d] = pil_to_array(jpeg_img)

            try:
                psnr_val = psnr(mri, mri_reconstruction)
            except:
                continue
        else:
            if dataset_name == "era5":
                img = ds[i][-1]  # Grayscale image
                img = Image.fromarray(np.uint8(img * 255.0), "L")
            elif dataset_name == "cifar10":
                img = ds[i][0]  # Discard label (img is PIL image)
            elif dataset_name == "kodak":
                img = Image.fromarray(ds[i])

            jpeg_img, num_bytes = SaveWithTargetSize(img, target_in_bytes, format)
            if dataset_name in ["fastmri", "era5"]:  # 💩 hack for greyscale
                jpeg_img = ImageOps.grayscale(jpeg_img)

            try:
                psnr_val = psnr(pil_to_array(img), pil_to_array(jpeg_img))
            except:
                continue

        psnr_vals.append(psnr_val)

    mean_psnr_val = sum(psnr_vals) / len(psnr_vals)
    print(f"\t{bpp}: {mean_psnr_val} dB")


if __name__ == "__main__":

    bpp_dict = {
        "cifar10": {
            "jpeg2000": [3.0, 3.8, 4.6, 5.4, 6.2, 7.0, 7.8],
            "bpg": [1.0, 1.25, 1.875, 2.5, 3.75, 5.0],
        },
        "era5": {
            "jpeg2000": [0.4, 0.6, 0.8, 1.0, 1.2],
            "bpg": [0.15, 0.2, 0.4, 0.7, 1.0],
        },
        "fastmri": {
            "jpeg2000": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
            "bpg": [0.03, 0.05, 0.075, 0.1, 0.15],
        },
    }

    for dataset_name in ["cifar10", "era5", "fastmri"]:
        for format in ["jpeg2000", "bpg"]:
            print(f"Dataset: {dataset_name} - Format: {format}")
            for bpp in bpp_dict[dataset_name][format]:
                print(f"\tbpp: {bpp}")
                try:
                    run(dataset_name, bpp, format)
                except:
                    print(f"\t🥲")
