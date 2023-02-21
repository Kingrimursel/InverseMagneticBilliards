import torch
from torch.utils.data import DataLoader


def train_model(model, train_dataset, validation_dataset, loss_fn, num_epochs, batch_size=128, filename=None):
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

    # TODO: GPU

    # Iterate over training epochs
    for epoch in range(num_epochs):
        # Track training loss
        epoch_train_loss = 0.0
        epoch_validation_loss = 0.0

        # Train the model for one epoch
        for inputs, targets in train_loader:
            # Reset the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            loss = loss_fn(outputs, targets)

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            # Track training loss
            epoch_train_loss += loss.item()

        # calculate validation loss
        for inputs, targets in validation_loader:
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            # Update validation loss
            epoch_validation_loss += loss.item()

        # Calculate average losses
        train_loss = epoch_train_loss / len(train_dataset)
        validation_loss = epoch_validation_loss / len(validation_dataset)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(
            epoch+1, train_loss, validation_loss))

    if filename:
        torch.save({
            "model_state_dict": model.state_dict(),
            "epochs": num_epochs,
            "train_losses": train_losses,
            "validation_losses": validation_losses
        },
            filename)

    return model, train_losses, validation_losses
