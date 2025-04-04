import numpy as np
import matplotlib.pyplot as plt

def show_image(img: np.ndarray, title: str = "Image") -> None:
    """
    Show an image using matplotlib.
    
    Parameters:
        img (np.ndarray): The image to show.
        title (str): The title of the image.
    """
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()