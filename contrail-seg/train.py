# %%
import os
import warnings
import click
import lightning
import torch
from torch.utils.data import DataLoader
from contrail import ContrailModel
import data

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


# if __name__ == "__main__":
#     main()

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

class PredictionCallback(Callback):
    def __init__(self, val_samples, output_dir="predictions", num_images=4):
        super().__init__()
        self.val_samples = val_samples
        self.output_dir = output_dir
        self.num_images = num_images
        os.makedirs(output_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
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

        # save side-by-side plots
        for i in range(self.num_images):
            fig, axs = plt.subplots(1, 3, figsize=(9, 3))
            axs[0].imshow(inputs[i, 0], cmap="gray")
            axs[0].set_title("Input")
            axs[1].imshow(targets[i, 0], cmap="gray")
            axs[1].set_title("Target")
            axs[2].imshow(preds[i, 0] > 0.5, cmap="gray")
            axs[2].set_title("Prediction")
            for ax in axs:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/epoch{trainer.current_epoch:03d}_sample{i}.png")
            plt.close(fig)


def train_contrail_network(augmentation, epochs, loss, base):

    print(
        f"training:  {epochs} epoch, {loss} loss, {base} base"
    )

    torch.cuda.empty_cache()
    
    # train_dataset, val_dataset = data.study_dataset(augmentation=augmentation)
    train_dataset, val_dataset = data.own_dataset_2(augmentation)


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

    prediction_callback = PredictionCallback(val_samples, output_dir="data/predictions")

    trainer = lightning.Trainer(
        max_epochs=epochs,
        log_every_n_steps=10,
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
        f_out = f"data/models/{loss}-{max_val}{tag}.torch"
    else:
        f_out = f"data/models/{loss}:{base}-{max_val}{tag}.torch"

    torch.save(model.state_dict(), f_out)