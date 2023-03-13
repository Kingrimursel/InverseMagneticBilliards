import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from util import batch_hessian, mkdir


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = torch.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_model(model,
                train_dataset,
                validation_dataset,
                loss_fn,
                num_epochs,
                batch_size=128,
                dir=None,
                device="cpu"):

    # move model to device
    model = model.to(device)

    # initialize early stopper
    early_stopper = EarlyStopper(patience=50, min_delta=1e-5)

    # Set model to training mode
    model.train()

    # Track loss
    train_losses = []
    validation_losses = []

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    train_loss = 0.
    validation_loss = 0.

    # Iterate over training epochs
    for epoch in (pbar := tqdm(range(num_epochs))):
        # Track training loss
        epoch_train_loss = 0.0
        epoch_validation_loss = 0.0

        pbar.set_postfix({'Training Loss': train_loss,
                         'Validation Loss': validation_loss})

        # Train the model for one epoch
        for inputs, targets in train_loader:

            # move to gpu
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Reset the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # required if output_dim=1, since output is of shape (output_dim, 1) then
            outputs = torch.squeeze(outputs)

            # Calculate main loss
            total_loss = loss_fn(outputs, targets)

            # backward pass
            total_loss.backward()

            # update parameters
            optimizer.step()

            # track training losses
            epoch_train_loss += total_loss.item()

        with torch.no_grad():
            # calculate validation loss
            for inputs, targets in validation_loader:

                # move to device
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(inputs)
                outputs = torch.squeeze(outputs)

                loss = loss_fn(outputs, targets)

                # Update validation loss
                epoch_validation_loss += loss.item()

        # Calculate average losses
        train_loss = epoch_train_loss / len(train_dataset)
        validation_loss = epoch_validation_loss / len(validation_dataset)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        # save model after each epoch
        if dir:
            mkdir(os.path.join(dir, "epochs", str(epoch+1)))

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch+1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            },
                os.path.join(dir, "epochs", str(epoch+1), "model.pth"))

        if early_stopper.early_stop(epoch_validation_loss):
            break

    # save final model
    if dir:
        torch.save({
            "model_state_dict": model.state_dict(),
            "epochs": epoch + 1,
            "train_losses": train_losses,
            "validation_losses": validation_losses,
        },
            os.path.join(dir, "model.pth"))

    return model, train_losses, validation_losses
