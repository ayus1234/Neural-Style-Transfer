import os
import requests
from PIL import Image
from io import BytesIO

def download_image(url, filename):
    """Download an image from URL and save it."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; NeuralStyleTransferBot/1.0)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        img = Image.open(BytesIO(response.content))
        img.save(filename)
        print(f"Downloaded {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {str(e)}")

def main():
    # Create images directory if it doesn't exist
    if not os.path.exists('images'):
        os.makedirs('images')

    # Sample images (using public domain images from Wikimedia Commons)
    images = {
        'content': 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Tuebingen_Neckarfront.jpg/640px-Tuebingen_Neckarfront.jpg',
        'style': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/640px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg'
    }

    # Download images
    for name, url in images.items():
        filename = f'images/{name}.jpg'
        download_image(url, filename)

if __name__ == "__main__":
    main() 