import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def trainer(model, optimizer, datamodule, num_epochs=10, device=device):

    # Move the model to the specified device
    model = model.to(device)

    # Define the loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Loop over epochs
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Train for one epoch
        train_loss = 0.0
        train_acc = 0.0
        model.train()
        for i, (inputs, labels) in enumerate(datamodule.train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_acc += torch.sum(preds == labels.data)
            
        # Print training loss and accuracy
        train_loss /= len(datamodule.train_dataloader)
        train_acc /= len(datamodule.train_dataloader.dataset)
        print(f'Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}')

        # Evaluate on validation set
        val_loss = 0.0
        val_acc = 0.0
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(datamodule.val_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_acc += torch.sum(preds == labels.data)
        
        # Print validation loss and accuracy
        val_loss /= len(datamodule.val_dataloader)
        val_acc /= len(datamodule.val_dataloader.dataset)
        print(f'Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}')
