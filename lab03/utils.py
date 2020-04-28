import matplotlib.pyplot as plt


def show_images_random(images_arr, num_of_images):
    figure = plt.figure()
    for index in range(1, num_of_images + 1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(images_arr[index].numpy().squeeze(), cmap='gray_r')
    plt.show()


def show_image(image):
    plt.imshow(image.numpy().squeeze(), cmap='gray_r')
    plt.show()
