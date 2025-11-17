# Quick Start Guide

Get started with the Image-to-3D Pipeline in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning)
- MiDaS model (depth estimation)
- Open3D & Trimesh (3D processing)
- Rembg (background removal)
- Image processing tools

**Note**: Installation may take 5-15 minutes depending on your internet speed.

## Step 2: Prepare Your Images

Place your product photos in the `input/` folder:

```bash
cp your_product_photo.jpg input/
```

**Best practices**:
- Clear, well-lit photos
- Object centered in frame
- Any background (will be removed automatically)
- Minimum 800x800px recommended

## Step 3: Run the Pipeline

### Single Image
```bash
python image_to_3d.py --input input/your_photo.jpg
```

### Batch Process All Images
```bash
python image_to_3d.py --batch
```

## Step 4: Check Your Results

Your outputs will be in:
- `output/stl/` - STL files ready for slicing
- `output/previews/` - Preview images
- `output/reports/` - Product metadata (cost, dimensions, etc.)

## Step 5: Print Your Model

1. Open the STL file in your slicer (Cura, PrusaSlicer, etc.)
2. Check the metadata JSON for recommended settings:
   - Print time estimate
   - Material cost
   - Dimensions
3. Slice and print!

## Common Commands

```bash
# Process with category
python image_to_3d.py --input photo.jpg --category figurine

# Batch process decorations
python image_to_3d.py --batch --category decoration

# View help
python image_to_3d.py --help
```

## Testing

Create a test image:
```bash
python create_test_image.py
python image_to_3d.py --input input/test_sphere.png
```

## Expected Processing Time

With GPU (recommended):
- Small image (512x512): ~30-40 seconds
- Large image (1024x1024): ~50-70 seconds

With CPU:
- Small image: ~2-3 minutes
- Large image: ~4-6 minutes

## Need Help?

- Check `logs/pipeline.log` for detailed information
- Read the full README.md for advanced features
- Adjust settings in `config.yaml`

## Next Steps

1. Experiment with different images
2. Try different product categories
3. Adjust quality settings in config.yaml
4. Set up your pricing in config.yaml
5. Process your entire product catalog in batch!

Happy printing! 🎨🖨️
