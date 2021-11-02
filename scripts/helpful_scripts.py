from PIL import Image
import tensorflow as tf
import matplotlib.pylab as plt
from matplotlib import gridspec
import os

# CONTENT_IMG_URL = (
#     "/content/drive/Shared drives/ever2/neural_style_transfer/content/glass.PNG"
# )
# STYLE_IMG_URL = "/content/drive/Shared drives/ever2/neural_style_transfer/style/Living-with-Lupus.PNG"
CONTENT_IMG_URL = ("./content/small_flower.jpg")
STYLE_IMG_URL = ("./style/style_duchamp.jpg")

def crop_center(image):
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape
    )
    return image


# @functools.lru_cache(maxsize=None)
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
    """Loads and preprocesses images."""
    # Cache image file locally.
    image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    img = tf.io.decode_image(tf.io.read_file(image_path), channels=3, dtype=tf.float32)[
        tf.newaxis, ...
    ]
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img


def show_n(images, titles=("",)):
    n = len(images)
    image_sizes = [image.shape[1] for image in images]
    w = (image_sizes[0] * 6) // 320
    plt.figure(figsize=(w * n, w))
    gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
    for i in range(n):
        plt.subplot(gs[i])
        plt.imshow(images[i][0], aspect="equal")
        plt.axis("off")
        plt.title(titles[i] if len(titles) > i else "")
    plt.show()


def load_local_img(image_path, image_size=(256, 256), preserve_aspect_ratio=True):
    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    img = tf.io.decode_image(
      tf.io.read_file(image_path), channels=3, dtype=tf.float32)[
        tf.newaxis, ...
    ]
    print("success decoded")
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    print("success resized")
    return img


# retrieve the converted img path
def convert_img_type(image_path):
    img_id = 0
    if "./content" in image_path:
        img_id += 1
        img_source = "converted_content_"
        im = Image.open(image_path)
        converted_path = f"./converted_content/converted_content_{img_id}.jpg"
        im.convert("RGB").save(
            converted_path, "JPEG"
        )  # this converts png image as jpeg
        return converted_path
    if "./style" in image_path:
        img_id += 1
        img_source = "converted_style_"
        im = Image.open(image_path)
        converted_path = f"./converted_style/converted_style_{img_id}.jpg"
        im.convert("RGB").save(
            converted_path, "JPEG"
        )  # this converts png image as jpeg
        return converted_path


def get_content_image_url(content_image_url):
    content_image_url = convert_img_type(content_image_url)
    print(content_image_url)
    return content_image_url


def get_style_image_url(style_image_url):
    style_image_url = convert_img_type(style_image_url)
    print(style_image_url)
    return style_image_url
