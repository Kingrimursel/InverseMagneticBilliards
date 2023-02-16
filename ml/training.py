import torch
from torch.utils.data import DataLoader

def train_model(model, train_data, loss_fn, num_epochs, batch_size=128):
    # Set model to training mode
    model.train()

    # Track loss
    train_losses = []

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    # TODO: GPU
    # TODO: use dataloaders?!

    # Iterate over training epochs
    for epoch in range(num_epochs):
        # Track training loss
        epoch_train_loss = 0.0

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

        # Calculate average losses
        train_loss = epoch_train_loss / len(train_data)

        train_losses.append(train_loss)

        print('Epoch: {}, Training Loss: {:.4f}'.format(epoch+1, train_loss))

    return model, train_losses
