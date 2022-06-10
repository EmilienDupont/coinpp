import random
import torch
import torchaudio


class LIBRISPEECH(torchaudio.datasets.LIBRISPEECH):
    """LIBRISPEECH dataset without labels.

    Args:
        patch_shape (int): Shape of patch to use. If -1, uses all data (no patching).
        num_secs (float): Number of seconds of audio to use. If -1, uses all available
            audio.
        normalize (bool): Whether to normalize data to lie in [0, 1].
    """

    def __init__(
        self,
        patch_shape: int = -1,
        num_secs: float = -1,
        normalize: bool = True,
        *args,
        **kwargs
    ):
        # TODO(emi): We should manually check if root exists, otherwise we should create
        # a directory. Somehow LIBRISPEECH does not do this automatically

        super().__init__(*args, **kwargs)

        # LibriSpeech contains audio 16kHz rate
        self.sample_rate = 16000

        self.normalize = normalize
        self.patch_shape = patch_shape
        self.random_crop = patch_shape != -1
        self.num_secs = num_secs
        self.num_waveform_samples = int(self.num_secs * self.sample_rate)

    def __getitem__(self, index):
        # __getitem__ returns a tuple, where first entry contains raw waveform in [-1, 1]
        datapoint = super().__getitem__(index)[0].float()

        # Normalize data to lie in [0, 1]
        if self.normalize:
            datapoint = (datapoint + 1) / 2

        # Extract only first num_waveform_samples from waveform
        if self.num_secs != -1:
            # Shape (channels, num_waveform_samples)
            datapoint = datapoint[:, : self.num_waveform_samples]

        if self.random_crop:
            datapoint = random_crop1d(datapoint, self.patch_shape)

        return datapoint


def random_crop1d(data, patch_shape: int):
    if not (0 < patch_shape <= data.shape[-1]):
        raise ValueError("Invalid shapes.")
    width_from = random.randint(0, data.shape[-1] - patch_shape)
    return data[
        ...,
        width_from : width_from + patch_shape,
    ]
