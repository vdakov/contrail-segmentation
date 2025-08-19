from matplotlib import pyplot as plt


def visualize_pytorch_dataset_images(trainset, title):
    fig, axs = plt.subplots(1, 10, figsize=(15, 3))
    for i in range(10):
        img, label = trainset[i]
        img = img.cpu().numpy().squeeze()  # Convert to numpy and remove channel dimension if present
        axs[i].imshow(img.squeeze(), cmap='gray')
        axs[i].set_title(f'Label: {label}')
        axs[i].axis('off')
    plt.suptitle(title)
    plt.show()