import matplotlib.pyplot as plt


def show_batch(images, labels, classes):

    for i in range(len(images)):
        image = images[i].detach().cpu().permute(1, 2, 0).numpy()
        label = labels[i].item()

        plt.imshow(image)
        plt.title(f"Label: {classes[label]}")
        plt.axis("off")
        plt.show()