import os
import subprocess
import time

def run_style_transfer():
    """Run the neural style transfer with sample images."""
    content_path = os.path.join('images', 'content.jpg')
    style_path = os.path.join('images', 'style.jpg')

    # Check if images exist
    if not os.path.exists(content_path) or not os.path.exists(style_path):
        print("Sample images not found. Please run download_sample_images.py first.")
        return

    # Run style transfer with different parameters
    print("\nRunning style transfer with default parameters...")
    subprocess.run([
        'python', 'neural_style_transfer.py',
        '--content', content_path,
        '--style', style_path
    ])

    print("\nRunning style transfer with more iterations and stronger style...")
    subprocess.run([
        'python', 'neural_style_transfer.py',
        '--content', content_path,
        '--style', style_path,
        '--iterations', '100',
        '--style-weight', '2000'
    ])

if __name__ == "__main__":
    # First, download sample images
    print("Downloading sample images...")
    subprocess.run(['python', 'download_sample_images.py'])
    
    # Then run style transfer
    print("\nStarting style transfer tests...")
    run_style_transfer() 