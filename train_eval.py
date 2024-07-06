import torch


# Training loop
def train_model(model, dataloader, epochs, optimizer, loss_fn, device):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            loss = loss_fn(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")

        # Clear any cached GPU memory
        torch.cuda.empty_cache()


def evaluate_model(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    total, correct = 0, 0
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)  # Ensure data and targets are on the correct device
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total  # Calculate accuracy as a percentage
    print(f"Total Correct: {correct}")
    print(f"Total Samples: {total}")
    print(f"Accuracy: {accuracy}%")
