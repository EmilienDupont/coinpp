import numpy as np
import subprocess
import torch
import torchaudio
from data.audio import LIBRISPEECH
from tqdm import tqdm


# Fill out location of librispeech dataset on your machine
LIBRISPEECH_ROOT = ""


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    if mse == 0:
        raise ValueError
    return -10.0 * np.log10(mse)


def run(format: str):
    ds = LIBRISPEECH(
        download=False,
        root=LIBRISPEECH_ROOT,
        url="test-clean",
        normalize=False,
        num_secs=3,
    )

    bit_rates = [20, 28, 36, 44, 52, 60, 72, 88, 124]

    out_psnr = np.zeros_like(bit_rates)

    for raw_input in tqdm(ds):
        # Save audio to file so it can be compressed with mp3
        torchaudio.save("in.wav", raw_input, sample_rate)

        for i, bit_rate in enumerate(bit_rates):
            # The MP3 lame encoder adds a delay of 576 samples depending on the bit rate
            delay = 576 if format == "mp3" and bit_rate < 37 else 0

            if format == "mp3":
                subprocess.run(
                    [
                        "lame",
                        "-b",
                        str(bit_rate),
                        "in.wav",
                        f"out.{format}",
                    ],
                    stderr=subprocess.DEVNULL,
                )

                subprocess.run(
                    [
                        "lame",
                        "--decode",
                        f"out.{format}",
                        "out.wav",
                    ],
                    stderr=subprocess.DEVNULL,
                )

            enc, sr = torchaudio.load(f"out.wav", normalize=True)

            if format == "mp3":
                # MP3 lame encoder adds a delay at the beginning and at the end of the
                # audio file, so remove these before computing psnr
                enc = enc[:, delay:]
                enc = enc[:, : len(raw_input[0])]

            assert sr == ds.sample_rate

            # psnr() assumes normalized values [0, 1], but enc is in [-1, 1], so
            # normalize
            psnr_val = psnr((raw_input + 1) / 2, (enc + 1) / 2)
            out_psnr[i] += psnr_val

    for idx, bit_rate in enumerate(bit_rates):
        print(f"Bit Rate: {bit_rate}")
        print(f"PSNR: {out_psnr[idx] / len(ds)}")


if __name__ == "__main__":

    for format in ["mp3"]:
        print(f" Evaluating {format} ".center(50, "="))
        run(format)
