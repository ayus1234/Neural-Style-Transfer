import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import argparse
import os

from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

def load_and_process_image(image_path):
    """Load and preprocess image for VGG19 model."""
    img = load_img(image_path)
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def deprocess(img):
    """Convert processed image back to displayable format."""
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]  # Convert BGR to RGB
    img = np.clip(img, 0, 255).astype('uint8')
    return img

def display_image(image):
    """Display the image."""
    if len(image.shape) == 4:
        img = np.squeeze(image, axis=0)
    img = deprocess(img)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    return

def gram_matrix(A):
    """Calculate Gram matrix for style representation."""
    channels = int(A.shape[-1])
    a = tf.reshape(A, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def content_loss(content, generated):
    """Calculate content loss between content and generated images."""
    a_C = content_model(content)
    a_G = content_model(generated)
    loss = tf.reduce_mean(tf.square(a_C - a_G))
    return loss

def style_cost(style, generated):
    """Calculate style loss between style and generated images."""
    J_style = 0
    for style_model in style_models:
        a_S = style_model(style)
        a_G = style_model(generated)
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)
        style_cost = tf.reduce_mean(tf.square(GS - GG))
        J_style += style_cost * weight_of_layer
    return J_style

def content_cost(style, generated):
    """Calculate content cost between style and generated images."""
    J_content = 0
    for style_model in style_models:
        a_S = style_model(style)
        a_G = style_model(generated)
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)
        content_cost = tf.reduce_mean(tf.square(GS - GG))
        J_content += content_cost * weight_of_layer
    return J_content

def training_loop(content_path, style_path, iterations=50, a=10, b=1000):
    """Main training loop for neural style transfer."""
    content = load_and_process_image(content_path)
    style = load_and_process_image(style_path)
    generated = tf.Variable(content, dtype=tf.float32)

    opt = tf.keras.optimizers.Adam(learning_rate=0.7)

    best_cost = math.inf
    best_image = None
    generated_images = []

    for i in range(iterations):
        start_time_cpu = time.process_time()
        start_time_wall = time.time()

        with tf.GradientTape() as tape:
            J_content = content_cost(style, generated)
            J_style = style_cost(style, generated)
            J_total = a * J_content + b * J_style

        grads = tape.gradient(J_total, generated)
        opt.apply_gradients([(grads, generated)])

        end_time_cpu = time.process_time()
        end_time_wall = time.time()
        cpu_time = end_time_cpu - start_time_cpu
        wall_time = end_time_wall - start_time_wall

        if J_total < best_cost:
            best_cost = J_total
            best_image = generated.numpy()

        print(f"CPU times: user {int(cpu_time * 1e6)} µs, sys: {int((end_time_cpu - start_time_cpu) * 1e9)} ns, total: {int((end_time_cpu - start_time_cpu + 1e-6) * 1e6)} µs")
        print(f"Wall time: {wall_time * 1e6:.2f} µs")
        print(f"Iteration: {i}")
        print(f'Total Loss {J_total:e}.')
        generated_images.append(generated.numpy())

    return best_image, generated_images

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('--content', type=str, required=True,
                      help='Path to content image')
    parser.add_argument('--style', type=str, required=True,
                      help='Path to style image')
    parser.add_argument('--iterations', type=int, default=50,
                      help='Number of optimization steps')
    parser.add_argument('--content-weight', type=float, default=10,
                      help='Weight for content loss')
    parser.add_argument('--style-weight', type=float, default=1000,
                      help='Weight for style loss')
    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.content):
        raise FileNotFoundError(f"Content image not found: {args.content}")
    if not os.path.exists(args.style):
        raise FileNotFoundError(f"Style image not found: {args.style}")

    # Initialize VGG19 model
    model = VGG19(include_top=False, weights='imagenet')
    model.trainable = False

    # Define content and style layers
    global content_model, style_models, weight_of_layer
    content_layer = 'block5_conv2'
    content_model = Model(inputs=model.input, outputs=model.get_layer(content_layer).output)

    style_layers = ['block1_conv1', 'block3_conv1', 'block5_conv1']
    style_models = [Model(inputs=model.input, outputs=model.get_layer(layer).output) for layer in style_layers]
    weight_of_layer = 1. / len(style_models)

    # Load and display content and style images
    content_img = load_and_process_image(args.content)
    style_img = load_and_process_image(args.style)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    display_image(content_img)
    plt.title('Content Image')
    plt.subplot(1, 2, 2)
    display_image(style_img)
    plt.title('Style Image')
    plt.show()

    # Run style transfer
    final_img, generated_images = training_loop(
        args.content, 
        args.style,
        iterations=args.iterations,
        a=args.content_weight,
        b=args.style_weight
    )

    # Display results
    plt.figure(figsize=(12, 12))
    for i in range(min(10, len(generated_images))):
        plt.subplot(4, 3, i + 1)
        display_image(generated_images[i+39])
    plt.show()

    plt.figure(figsize=(10, 10))
    display_image(final_img)
    plt.title('Final Result')
    plt.show()

if __name__ == "__main__":
    main() 