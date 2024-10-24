import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('thiennhien.jpg', cv2.IMREAD_GRAYSCALE)

def show_images(images, titles, cmap='gray'):
    plt.figure(figsize=(10, 8))
    for i in range(len(images)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

def negative_image(img):
    return 255 - img

def contrast_enhancement(img):
    min_val = np.min(img)
    max_val = np.max(img)
    enhanced_img = 255 * (img - min_val) / (max_val - min_val)
    return enhanced_img.astype(np.uint8)

def log_transformation(img):
    img = img / 255.0  
    c = 255 / np.log(1 + np.max(img)) 
    log_img = c * np.log(1 + img)
    return np.array(log_img, dtype=np.uint8)

def histogram_equalization(img):
    return cv2.equalizeHist(img)

negative_img = negative_image(image)
contrast_img = contrast_enhancement(image)
log_img = log_transformation(image)
hist_eq_img = histogram_equalization(image)

show_images(
    [image, negative_img, contrast_img, log_img, hist_eq_img],
    ['Original Image', 'Negative Image', 'Contrast Enhanced', 'Log Transformed', 'Histogram Equalization']
)
