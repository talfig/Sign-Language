import torch
import matplotlib.pyplot as plt


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
            # Clear any cached GPU memory
            torch.cuda.empty_cache()

    accuracy = 100 * correct / total  # Calculate accuracy as a percentage
    print(f"Total Correct: {correct}")
    print(f"Total Samples: {total}")
    print(f"Accuracy: {accuracy}%")


def predict_and_display(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode

    # Define the grid size for 26 images (5 rows x 6 columns)
    nrows, ncols = 5, 6
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 12))
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    img_count = 0  # Counter for the number of images displayed

    with torch.no_grad():
        for images, labels in dataloader:
            for image, true_label in zip(images, labels):
                if img_count >= nrows * ncols:
                    break  # Stop if we have displayed 26 images

                # Ensure the image is on the correct device and add a batch dimension
                image = image.to(device).unsqueeze(0)

                # Perform the prediction
                outputs = model(image)
                _, predicted = torch.max(outputs.data, 1)

                # Convert to letter (0 -> A, 1 -> B, ..., 25 -> Z)
                predicted_label = chr(predicted.item() + ord('A'))
                # Convert real label to letter
                true_label_str = chr(true_label.item() + ord('A'))

                # Convert image to numpy array and squeeze to remove the batch dimension
                image_np = image.squeeze().cpu().numpy()

                # Display the image and prediction
                ax = axes[img_count]
                ax.imshow(image_np, cmap='gray')  # Display the image
                ax.set_title(f'True: {true_label_str}\nPred: {predicted_label}')  # Show both real and predicted labels
                ax.axis('off')  # Hide the axis

                img_count += 1

            if img_count >= nrows * ncols:
                break  # Stop if we have displayed 26 images

    # Turn off any remaining axes if there are fewer than 26 images
    for ax in axes[img_count:]:
        ax.axis('off')  # Hide the axis for unused subplots

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
