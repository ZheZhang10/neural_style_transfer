# Copyright 2019 The TensorFlow Hub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import functools
import os

# from google.colab import drive

from helpful_scripts import (
    load_local_img,
    show_n,
    get_content_image_url,
    get_style_image_url,
    CONTENT_IMG_URL,
    STYLE_IMG_URL,
)
from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub

print("TF Version: ", tf.__version__)
print("TF-Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.config.list_physical_devices("GPU"))

# drive.mount('/content/drive/', force_remount=True)
# COLAB = True


# image input > convert image type to jpg > get new content and style image urls >load&processing images(decode , crop(), resize)
# >set image and retrieve images > run model and stylize



def set_img(content_image_url, style_image_url):
    output_image_size = 384
    # The content image size can be arbitrary.
    content_img_size = (output_image_size, output_image_size)
    # The style prediction model was trained with image size 256 and it's the
    # recommended image size for the style image (though, other sizes work as
    # well but will lead to different results).
    style_img_size = (256, 256)  # Recommended to keep it at 256.
    content_image_url = get_content_image_url(content_image_url)
    style_image_url = get_style_image_url(style_image_url)
    print("Succeed got img url!!!")
    content_image = load_local_img(content_image_url, content_img_size)
    style_image = load_local_img(style_image_url, style_img_size)
    style_image = tf.nn.avg_pool(
        style_image, ksize=[3, 3], strides=[1, 1], padding="SAME"
    )
    # show_n([content_image, style_image], ["Content image", "Style image"])
    print("Succeed set img!!!!")
    run_model_and_stylize(content_image, style_image)

def run_model_and_stylize(content_image, style_image):
    # Load TF-Hub module.
    # hub_handle = '/content/drive/Shared drives/ever2/neural_style_transfer/model'
    hub_handle = "./neural_style_transfer_model"
    hub_module = hub.load(hub_handle)
    print("Succeed load model!!")
    # Stylize content image with given style image.
    # This is pretty fast within a few milliseconds on a GPU.
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]
    # Visualize input images and the generated stylized image.
    show_n(
        [content_image, style_image, stylized_image],
        titles=["Original content image", "Style image", "Stylized image"],
    )


content_image_url = CONTENT_IMG_URL
style_image_url = STYLE_IMG_URL
set_img(content_image_url, style_image_url)



