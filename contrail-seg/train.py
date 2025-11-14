# %%
import os
import warnings
import click
import lightning
from matplotlib import gridspec
import torch
from torch.utils.data import DataLoader
from contrail import ContrailModel
import data
import matplotlib.pyplot as plt
import numpy as np
from lightning.pytorch.callbacks import Callback
from IPython.display import display, clear_output

warnings.filterwarnings("ignore")
@click.command()
@click.option("--dataset", required=True)
@click.option("--minute", required=False, type=int, help="minutes")
@click.option("--epoch", required=False, type=int, help="minutes")
@click.option("--loss", required=True, help="dice, focal, or sr")
@click.option("--base", required=False, help="dice or focal (for sr loss)")
def main(dataset, minute, epoch, loss, base):

    print(
        f"training: {dataset} data, {minute} minutes, {epoch} epoch, {loss} loss, {base} base"
    )

    torch.cuda.empty_cache()

    if dataset == "own":
        train_dataset, val_dataset = data.own_dataset()
    elif dataset == "google":
        train_dataset, val_dataset = data.google_dataset()
    elif dataset.startswith("google:fewshot:"):
        n = int(dataset.split(":")[-1])
        train_dataset, val_dataset = data.google_dataset_few_shot(n=n)
    else:
        print(f"dataset: {dataset} unknown")
        return

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
    )

    model = ContrailModel(arch="UNet", in_channels=1, out_classes=1, loss=loss)

    # callback_save_model = lightning.callbacks.ModelCheckpoint(
    #     dirpath="data/models/",
    #     filename="google-dice-{epoch:02d}epoch.torch",
    #     save_top_k=-1,
    #     every_n_epochs=10,
    # )

    if minute is not None:
        trainer = lightning.Trainer(
            max_time=f"00:{(minute//60):02d}:{(minute%60):02d}:00",
            log_every_n_steps=20,
        )
        max_val = minute
        tag = "minute"

    elif epoch is not None:
        trainer = lightning.Trainer(
            max_epochs=epoch,
            log_every_n_steps=20,
        )
        max_val = epoch
        tag = "epoch"

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    if base is None:
        f_out = f"data/models/{dataset}-{loss}-{max_val}{tag}.torch"
    else:
        f_out = f"data/models/{dataset}-{loss}:{base}-{max_val}{tag}.torch"

    torch.save(model.state_dict(), f_out)

class LossAndPredictionCallback(Callback):
    def __init__(self, val_samples, output_dir="predictions", num_images=4, plot_every=5, experiment_name="experiment"):
        super().__init__()
        self.val_samples = val_samples
        self.output_dir = output_dir
        self.num_images = num_images
        self.plot_every = plot_every
        self.train_losses = []
        self.val_losses = []
        self.experiment_name = experiment_name
        os.makedirs(output_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        # record training loss
        train_loss = trainer.logged_metrics.get("train_loss")  # assumes you log 'train_loss'
        if train_loss is not None:
            self.train_losses.append(train_loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        # record validation loss
        val_loss = trainer.logged_metrics.get("val_loss")  # assumes you log 'val_loss'
        if val_loss is not None:
            self.val_losses.append(val_loss.item())

        # visualize predictions
        pl_module.eval()
        device = pl_module.device

        inputs, targets = [], []
        for i, (x, y) in enumerate(self.val_samples):
            if i >= self.num_images:
                break
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y)
            inputs.append(x.unsqueeze(0))
            targets.append(y.unsqueeze(0))
        inputs = torch.cat(inputs).to(device)
        targets = torch.cat(targets).to(device)

        with torch.no_grad():
            preds = pl_module(inputs)

        preds = torch.sigmoid(preds).cpu().numpy()
        inputs = inputs.cpu().numpy()
        targets = targets.cpu().numpy()
        clear_output(wait=True)  # clears previous output

        # save side-by-side plots for each image


        # plot losses every `plot_every` epochs
        if (trainer.current_epoch + 1) % self.plot_every == 0:
            fig = plt.figure(figsize=(12, 6))
            gs = gridspec.GridSpec(2, self.num_images, height_ratios=[1, 0.5])

            # Top row: Loss curves
            ax0 = fig.add_subplot(gs[0, :])
            ax0.plot(range(1, len(self.train_losses)+1), self.train_losses, label="Train Loss")
            ax0.plot(range(1, len(self.val_losses)+1), self.val_losses, label="Val Loss")
            ax0.set_xlabel("Epoch")
            ax0.set_ylabel("Loss")
            ax0.set_title(f"Epoch {trainer.current_epoch+1} - Loss Curves")
            ax0.legend()
            ax0.grid(True)

            # Bottom row: Predictions
            for i in range(self.num_images):
                ax = fig.add_subplot(gs[1, i])
                combined = np.concatenate(
                    [inputs[i, 0], targets[i, 0], (preds[i, 0] > 0.5).astype(float)], axis=1
                )
                ax.imshow(combined, cmap="gray")
                ax.axis("off")
                ax.set_title(f"Sample {i}")

            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/{self.experiment_name}-epoch{trainer.current_epoch+1:03d}_summary.png")
            display(plt.gcf())
            plt.close()



def train_contrail_network(augmentation, epochs, loss, base, dataset="own", log_every_n_epochs=5, experiment_name="experiment"):

    print(
        f"training:  {epochs} epoch, {loss} loss, {base} base"
    )

    torch.cuda.empty_cache()
    
    train_dataset, val_dataset = None, None
    
    if dataset == "own":
        train_dataset, val_dataset = data.own_dataset_2(augmentation)
    elif dataset == "google":
        train_dataset, val_dataset = data.google_dataset(augmentation=augmentation)


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
    )

    model = ContrailModel(arch="UNet", in_channels=1, out_classes=1, loss=loss)

    # pick a few samples for visualization
    val_samples = [val_dataset[i] for i in range(4)]

    prediction_callback = LossAndPredictionCallback(val_samples, output_dir="data/predictions")

    trainer = lightning.Trainer(
        max_epochs=epochs,
        log_every_n_steps=log_every_n_epochs,
        callbacks=[prediction_callback],
    )
    
    max_val = epochs
    tag = "epoch"

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    os.makedirs("data/models", exist_ok=True)


    if base is None:
        f_out = f"data/models/{experiment_name}-{dataset}-{loss}-{max_val}{tag}.torch"
    else:
        f_out = f"data/models/{experiment_name}-{dataset}-{loss}:{base}-{max_val}{tag}.torch"


    torch.save(model.state_dict(), f_out)