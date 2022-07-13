# COIN++

Official implementation of [COIN++: Neural Compression Across Modalities](https://arxiv.org/abs/2201.12904).

<img src="https://github.com/EmilienDupont/coinpp/raw/main/imgs/fig1.png" width="800">

## Requirements

The requirements can be found in `requirements.txt`. While it is possible to run most of the code without it, we *strongly* recommend using [wandb](https://wandb.ai/) for experiment logging and storing as this is tighly integrated with the codebase.

## Data

Before running experiments, make sure to set data paths in `data/dataset_paths.yml`. Most datasets can be downloaded automatically, except for [FastMRI](https://fastmri.org/) which needs an application form and ERA5 which can be downloaded [here](https://github.com/EmilienDupont/neural-function-distributions#downloading-datasets). For the FastMRI dataset, we use the `brain_multicoil_val.zip` file and split into train and test sets using the ids in `data/fastmri_split_ids.py`.

## Training

To train a model, run

```python main.py @config.txt```.

See `config.txt` and `main.py` for setting various arguments. Note that if using wandb, you need to change the wandb entity and project name to your own.

A few example configs used to train the models in the paper can be found in the `configs` folder.

#### Storing modulations

Given the `wandb_run_path` from a trained model, store modulations using

```python store_modulations --wandb_run_path <wandb_run_path>```.

#### Evaluation

To evaluate the performance of a given modulation dataset (in terms of PSNR), run

```python evaluate.py --wandb_run_path <wandb_run_path> --modulation_dataset <modulation_dataset>```.

#### Quantization

To quantize a modulation dataset to a given bitwidth, run

```python quantization.py --wandb_run_path <wandb_run_path> --train_mod_dataset <train_mod_dataset> --test_mod_dataset <test_mod_dataset> --num_bits 5```.

#### Entropy coding

To entropy code a quantized modulation dataset, run

```python entropy_coding.py --wandb_run_path <wandb_run_path> --train_mod_dataset <train_mod_dataset> --test_mod_dataset <test_mod_dataset>```.

#### Saving reconstructions

To save reconstructions for a specific set of data points, run

```python reconstruction.py --wandb_run_path <wandb_run_path> --modulation_dataset <modulation_dataset> --data_indices 0 1 2 3```.

## Trained models and modulations [Not yet public ⚠️]

_The trained models, runs and modulations are not yet public as we need to share wandb runs from a private project (see this [github issue](https://github.com/wandb/client/issues/3764)). We hope to make this public soon!_

All models and modulations are stored on wandb. To find the link for a given model or run, see the `wandb_ids.json` files in the appropriate folder in the `results` directory. The model and run information can the be found at `wandb.ai/<wandb_id>`.

## Results and plots

To recreate all the plots in the paper run:

```python plots.py```.

See `plots.py` for plotting options. All results and ablations can be found in the `results` folder.

## Baselines

Running the baselines requires that all codecs are installed on your machine. In addition, the baseline scripts also require `tqdm` and `PIL`.

#### Image baselines

The image baselines used for CIFAR10, Kodak, FastMRI and ERA5 are:
- JPEG: We use the JPEG implementation from PIL version 8.1.0.
- JPEG2000: We use the JPEG2000 implementation from OpenJPEG version 2.4.0.
- BPG: We use BPG version 0.9.8.

#### Audio baselines

The audio baseline used for LibriSpeech is:
- MP3: We use the MP3 implementation from LAME version 3.100.

## License

MIT