import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from util import batch_jacobian, batch_hessian
from dynamics import Orbit
from physics import DiscreteAction


def train_model(model,
                train_dataset,
                validation_dataset,
                loss_fn,
                num_epochs,
                batch_size=128,
                dir=None,
                alpha=1e-3):

    # Set model to training mode
    model.train()

    # Track loss
    train_losses = []
    validation_losses = []

    hess_train_losses = []

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Iterate over training epochs
    for epoch in range(num_epochs):
        # Track training loss
        epoch_train_loss = 0.0
        epoch_validation_loss = 0.0
        epoch_hess_train_loss = 0.0

        # Train the model for one epoch
        for inputs, targets in train_loader:
            # Reset the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # required if output_dim=1, since output is of shape (output_dim, 1) then
            outputs = torch.squeeze(outputs)

            # Calculate main loss
            loss = loss_fn(outputs, targets)

            # regularizer loss
            hessian = torch.squeeze(batch_hessian(model.model, inputs))
            hess_loss = torch.mean(hessian.pow(2))

            # total loss as weighted sum
            total_loss = loss + alpha*hess_loss

            # Backward pass
            total_loss.backward()

            # Update parameters
            optimizer.step()

            # Track training loss
            epoch_train_loss += loss.item()
            epoch_hess_train_loss += alpha*hess_loss.item()

        with torch.no_grad():
            # calculate validation loss
            for inputs, targets in validation_loader:
                # Forward pass
                outputs = model(inputs)
                outputs = torch.squeeze(outputs)

                loss = loss_fn(outputs, targets)

                # Update validation loss
                epoch_validation_loss += loss.item()

        # Calculate average losses
        train_loss = epoch_train_loss / len(train_dataset)
        hess_train_loss = epoch_hess_train_loss / len(train_dataset)
        validation_loss = epoch_validation_loss / len(validation_dataset)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        hess_train_losses.append(hess_train_loss)

        # save model
        if dir:
            Path(os.path.join(dir, "epochs", str(epoch+1))
                 ).mkdir(parents=True, exist_ok=True)

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch+1,
                "train_loss": train_loss,
                "validation_loss": validation_loss
            },
                os.path.join(dir, "epochs", str(epoch+1), "model.pth"))

        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}. Hessian Loss: {:.4f}'.format(
            epoch+1, train_loss, validation_loss, hess_train_loss))

    if dir:
        torch.save({
            "model_state_dict": model.state_dict(),
            "epochs": num_epochs,
            "train_losses": train_losses,
            "validation_losses": validation_losses
        },
            os.path.join(dir, "model.pth"))

    return model, train_losses, validation_losses, hess_train_losses
