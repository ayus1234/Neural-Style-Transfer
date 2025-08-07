# Neural Style Transfer

This project implements Neural Style Transfer using TensorFlow and the VGG19 model. It allows you to apply the artistic style of one image to the content of another image.

## Requirements

- Python 3.7+
- TensorFlow 2.4.0+
- NumPy
- Matplotlib
- Pillow

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the script with your content and style images:

```bash
python neural_style_transfer.py --content path/to/content.jpg --style path/to/style.jpg
```

Optional arguments:
- `--iterations`: Number of optimization steps (default: 50)
- `--content-weight`: Weight for content loss (default: 10)
- `--style-weight`: Weight for style loss (default: 1000)

Example with all options:
```bash
python neural_style_transfer.py \
    --content path/to/content.jpg \
    --style path/to/style.jpg \
    --iterations 100 \
    --content-weight 10 \
    --style-weight 1000
```

## How it Works

The implementation uses the VGG19 model to extract features from both the content and style images. The process involves:

1. Loading and preprocessing the content and style images
2. Using VGG19 to extract features from different layers
3. Computing content loss to preserve the content image's structure
4. Computing style loss using Gram matrices to capture the style image's texture
5. Optimizing the generated image to minimize both losses

## Output

The script will display:
1. The original content and style images
2. Progress updates during the style transfer process
3. The final stylized image
4. A grid of intermediate results

## Notes

- The process may take several minutes depending on your hardware.
- For best results, use images of similar sizes.
- Higher resolution images will require more memory and processing time.
- Supported image formats: JPG, PNG, BMP

## License
This project is licensed under the MIT License. See the LICENSE file for details.
