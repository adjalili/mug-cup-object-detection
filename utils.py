import matplotlib.pyplot as plt


def show_batch(images, labels, classes, n=8):
    """
    Display a batch of images in a grid
    """

    # Limit number of images
    images = images[:n]
    labels = labels[:n]

    plt.figure(figsize=(12, 6))

    for i in range(len(images)):
        image = images[i].detach().cpu().permute(1, 2, 0).numpy()

        # ✅ Fix potential normalization / clipping issues
        image = image.clip(0, 1)

        label = labels[i].item()

        plt.subplot(2, n // 2, i + 1)
        plt.imshow(image)
        plt.title(classes[label])
        plt.axis("off")

    plt.tight_layout()
    plt.show()