#!/bin/bash
# Example usage scripts for the Image-to-3D Pipeline

echo "Image-to-3D Pipeline - Usage Examples"
echo "====================================="
echo ""

# Example 1: Process a single image
echo "Example 1: Process a single product photo"
echo "Command: python image_to_3d.py --input input/my_product.jpg --category figurine"
echo ""

# Example 2: Batch process
echo "Example 2: Batch process all images in input folder"
echo "Command: python image_to_3d.py --batch --category decoration"
echo ""

# Example 3: Custom config
echo "Example 3: Use custom configuration"
echo "Command: python image_to_3d.py --config custom_config.yaml --batch"
echo ""

# Example 4: Different categories
echo "Example 4: Process different product categories"
echo "  Figurines: python image_to_3d.py --input photo.jpg --category figurine"
echo "  Decorations: python image_to_3d.py --input photo.jpg --category decoration"
echo "  Functional: python image_to_3d.py --input photo.jpg --category functional"
echo "  Prototypes: python image_to_3d.py --input photo.jpg --category prototype"
echo ""

echo "For more details, see README.md"
