import coinpp.conversion as conversion
import coinpp.losses as losses
import coinpp.metalearning as metalearning
import torch
import wandb


class Trainer:
    def __init__(
        self,
        func_rep,
        converter,
        args,
        train_dataset,
        test_dataset,
        patcher=None,
        model_path="",
    ):
        """Module to handle meta-learning of COIN++ model.

        Args:
            func_rep (models.ModulatedSiren):
            converter (conversion.Converter):
            args: Training arguments (see main.py).
            train_dataset:
            test_dataset:
            patcher: If not None, patcher that is used to create random patches during
                training and to partition data into patches during validation.
            model_path: If not empty, wandb path where best (validation) model
                will be saved.
        """
        self.func_rep = func_rep
        self.converter = converter
        self.args = args
        self.patcher = patcher

        self.outer_optimizer = torch.optim.Adam(
            self.func_rep.parameters(), lr=args.outer_lr
        )

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self._process_datasets()

        self.model_path = model_path
        self.step = 0
        self.best_val_psnr = 0.0

    def _process_datasets(self):
        """Create dataloaders for datasets based on self.args."""
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.num_workers > 0,
        )

        # If we are using patching, require data loader to have a batch size of 1,
        # since we can potentially have different sized outputs which cannot be batched
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=1 if self.patcher else self.args.batch_size,
            num_workers=self.args.num_workers,
        )

    def train_epoch(self):
        """Train model for a single epoch."""
        for data in self.train_dataloader:
            data = data.to(self.args.device)
            coordinates, features = self.converter.to_coordinates_and_features(data)

            # Optionally subsample points
            if self.args.subsample_num_points != -1:
                # Coordinates have shape (batch_size, *, coordinate_dim)
                # Features have shape (batch_size, *, feature_dim)
                # Flatten both along spatial dimension and randomly select points
                coordinates = coordinates.reshape(
                    coordinates.shape[0], -1, coordinates.shape[-1]
                )
                features = features.reshape(features.shape[0], -1, features.shape[-1])
                # Compute random indices (no good pytorch function to do this,
                # so do it this slightly hacky way)
                permutation = torch.randperm(coordinates.shape[1])
                idx = permutation[: self.args.subsample_num_points]
                coordinates = coordinates[:, idx, :]
                features = features[:, idx, :]

            outputs = metalearning.outer_step(
                self.func_rep,
                coordinates,
                features,
                inner_steps=self.args.inner_steps,
                inner_lr=self.args.inner_lr,
                is_train=True,
                return_reconstructions=False,
                gradient_checkpointing=self.args.gradient_checkpointing,
            )

            # Update parameters of base network
            self.outer_optimizer.zero_grad()
            outputs["loss"].backward(create_graph=False)
            self.outer_optimizer.step()

            if self.step % self.args.validate_every == 0 and self.step != 0:
                self.validation()

            log_dict = {"loss": outputs["loss"].item(), "psnr": outputs["psnr"]}

            self.step += 1

            print(
                f'Step {self.step}, Loss {log_dict["loss"]:.3f}, PSNR {log_dict["psnr"]:.3f}'
            )

            if self.args.use_wandb:
                wandb.log(log_dict, step=self.step)

    def validation(self):
        """Run trained model on validation dataset."""
        print(f"\nValidation, Step {self.step}:")

        # If num_validation_points is -1, validate on entire validation dataset,
        # otherwise validate on a subsample of points
        full_validation = self.args.num_validation_points == -1
        num_validation_batches = self.args.num_validation_points // self.args.batch_size

        # Initialize validation logging dict
        log_dict = {}

        # Evaluate model for different numbers of inner loop steps
        for inner_steps in self.args.validation_inner_steps:
            log_dict[f"val_psnr_{inner_steps}_steps"] = 0.0
            log_dict[f"val_loss_{inner_steps}_steps"] = 0.0

            # Fit modulations for each validation datapoint
            for i, data in enumerate(self.test_dataloader):
                data = data.to(self.args.device)
                if self.patcher:
                    # If using patching, test data will have a batch size of 1.
                    # Remove batch dimension and instead convert data into
                    # patches, with patch dimension acting as batch size
                    patches, spatial_shape = self.patcher.patch(data[0])
                    coordinates, features = self.converter.to_coordinates_and_features(
                        patches
                    )

                    # As num_patches may be much larger than args.batch_size,
                    # split the fitting of patches into batch_size chunks to
                    # reduce memory
                    outputs = metalearning.outer_step_chunked(
                        self.func_rep,
                        coordinates,
                        features,
                        inner_steps=inner_steps,
                        inner_lr=self.args.inner_lr,
                        chunk_size=self.args.batch_size,
                        gradient_checkpointing=self.args.gradient_checkpointing,
                    )

                    # Shape (num_patches, *patch_shape, feature_dim)
                    patch_features = outputs["reconstructions"]

                    # When using patches, we cannot directly use psnr and loss
                    # output by outer step, since these are calculated on the
                    # padded patches. Therefore we need to reconstruct the data
                    # in its original unpadded form and manually calculate mse
                    # and psnr
                    # Shape (num_patches, *patch_shape, feature_dim) ->
                    # (num_patches, feature_dim, *patch_shape)
                    patch_data = conversion.features2data(patch_features, batched=True)
                    # Shape (feature_dim, *spatial_shape)
                    data_recon = self.patcher.unpatch(patch_data, spatial_shape)
                    # Calculate MSE and PSNR values and log them
                    mse = losses.mse_fn(data_recon, data[0])
                    psnr = losses.mse2psnr(mse)
                    log_dict[f"val_psnr_{inner_steps}_steps"] += psnr.item()
                    log_dict[f"val_loss_{inner_steps}_steps"] += mse.item()
                else:
                    coordinates, features = self.converter.to_coordinates_and_features(
                        data
                    )

                    outputs = metalearning.outer_step(
                        self.func_rep,
                        coordinates,
                        features,
                        inner_steps=inner_steps,
                        inner_lr=self.args.inner_lr,
                        is_train=False,
                        return_reconstructions=True,
                        gradient_checkpointing=self.args.gradient_checkpointing,
                    )

                    log_dict[f"val_psnr_{inner_steps}_steps"] += outputs["psnr"]
                    log_dict[f"val_loss_{inner_steps}_steps"] += outputs["loss"].item()

                if not full_validation and i >= num_validation_batches - 1:
                    break

            # Calculate average PSNR and loss by dividing by number of batches
            log_dict[f"val_psnr_{inner_steps}_steps"] /= i + 1
            log_dict[f"val_loss_{inner_steps}_steps"] /= i + 1

            mean_psnr, mean_loss = (
                log_dict[f"val_psnr_{inner_steps}_steps"],
                log_dict[f"val_loss_{inner_steps}_steps"],
            )
            print(
                f"Inner steps {inner_steps}, Loss {mean_loss:.3f}, PSNR {mean_psnr:.3f}"
            )

            # Use first setting of inner steps for best validation PSNR
            if inner_steps == self.args.validation_inner_steps[0]:
                if mean_psnr > self.best_val_psnr:
                    self.best_val_psnr = mean_psnr
                    # Optionally save new best model
                    if self.args.use_wandb and self.model_path:
                        torch.save(
                            {
                                "args": self.args,
                                "state_dict": self.func_rep.state_dict(),
                            },
                            self.model_path,
                        )

            if self.args.use_wandb:
                # Store final batch of reconstructions to visually inspect model
                # Shape (batch_size, channels, *spatial_dims)
                reconstruction = self.converter.to_data(
                    None, outputs["reconstructions"]
                )
                if self.patcher:
                    # If using patches, unpatch the reconstruction
                    # Shape (channels, *spatial_dims)
                    reconstruction = self.patcher.unpatch(reconstruction, spatial_shape)
                if self.converter.data_type == "mri":
                    # To store an image, slice MRI data along a single dimension
                    # Shape (1, depth, height, width) -> (1, height, width)
                    reconstruction = reconstruction[:, reconstruction.shape[1] // 2]

                if self.converter.data_type == "audio":
                    # Currently only support audio saving when using patches
                    if self.patcher:
                        # Unnormalize data from [0, 1] to [-1, 1] as expected by wandb
                        if self.test_dataloader.dataset.normalize:
                            reconstruction = 2 * reconstruction - 1
                        # Saved audio sample needs shape (num_samples, num_channels),
                        # so transpose
                        log_dict[
                            f"val_reconstruction_{inner_steps}_steps"
                        ] = wandb.Audio(
                            reconstruction.T.cpu(),
                            sample_rate=self.test_dataloader.dataset.sample_rate,
                        )
                else:
                    log_dict[f"val_reconstruction_{inner_steps}_steps"] = wandb.Image(
                        reconstruction
                    )

                wandb.log(log_dict, step=self.step)

        print("\n")
