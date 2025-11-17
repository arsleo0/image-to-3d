# Image-to-3D Pipeline for Commercial 3D Printing

A production-ready pipeline that converts product photos into printable STL files with business analytics for 3D printing operations.

## Features

- **Smart Image Preprocessing**: Automatic background removal, centering, and optimization
- **High-Quality Depth Estimation**: Uses Intel's MiDaS DPT-Large model for accurate depth mapping
- **Robust Mesh Generation**: Poisson surface reconstruction with watertight guarantee
- **3D Printing Optimization**: Auto-scaling, orientation, and manifold checking
- **Business Analytics**: Cost calculation, pricing suggestions, and SKU generation
- **Batch Processing**: Process entire folders with progress tracking
- **Export & Preview**: STL files, multi-angle previews, and marketplace thumbnails

## Installation

1. Clone the repository and navigate to the project directory

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For HEIC support from iPhone photos:
```bash
pip install pillow-heif
```

## Quick Start

### Process a Single Image

```bash
python image_to_3d.py --input path/to/image.jpg --category figurine
```

### Batch Process All Images in Input Folder

```bash
python image_to_3d.py --batch --category decoration
```

## Usage

### Directory Structure

```
image-to-3d-pipeline/
├── input/              # Place your product photos here
├── processing/         # Intermediate files (depth maps, etc.)
├── output/
│   ├── stl/           # Final 3D printable STL files
│   ├── previews/      # Multi-angle preview renders
│   └── reports/       # JSON metadata for each product
├── logs/              # Pipeline execution logs
├── config.yaml        # Configuration settings
└── requirements.txt   # Python dependencies
```

### Configuration

Edit `config.yaml` to customize:

- **Image Resolution**: `target_resolution` (512 for speed, 1024 for quality)
- **Mesh Detail**: `simplify_faces` (10k-50k faces)
- **Print Size**: `max_dimension_mm` (default 100mm)
- **Material Cost**: `material_cost_per_gram` (default €0.02/gram)
- **Pricing**: `retail_markup` (default 5x material cost)

### Command Line Options

```
--config CONFIG      Path to config file (default: config.yaml)
--input IMAGE        Single image to process
--category CATEGORY  Product category (default/figurine/decoration/functional/prototype)
--batch              Process all images in input directory
```

### Product Categories

- `default`: General products (SKU prefix: GEN)
- `figurine`: Collectibles and miniatures (SKU prefix: FIG)
- `decoration`: Decorative items (SKU prefix: DEC)
- `functional`: Functional parts (SKU prefix: FUN)
- `prototype`: Prototypes and samples (SKU prefix: PRO)

## Output Files

For each processed image, the pipeline generates:

1. **STL File**: `output/stl/{SKU}.stl` - Ready for slicing
2. **Previews**: `output/previews/{SKU}_preview_{angle}deg.png` - 4 viewing angles
3. **Thumbnail**: `output/previews/{SKU}_thumb.png` - Square thumbnail for listings
4. **Metadata**: `output/reports/{SKU}.json` - Complete product information

### Metadata Includes

```json
{
  "sku": "3DP-GEN-20250117-1234",
  "volume_mm3": 15234.56,
  "weight_g": 18.89,
  "material_cost_eur": 0.38,
  "print_time_hours": 2.5,
  "recommended_price_eur": 1.90,
  "dimensions_mm": {"x": 95.2, "y": 87.3, "z": 100.0},
  "is_watertight": true,
  "vertex_count": 12547,
  "face_count": 24891
}
```

## Hardware Requirements

### Minimum
- CPU: Any modern multi-core processor
- RAM: 8GB
- Storage: 5GB for models + output

### Recommended
- GPU: NVIDIA GPU with 6GB+ VRAM (CUDA support)
- RAM: 16GB
- Storage: SSD with 20GB+ free space

The pipeline automatically detects GPU availability and falls back to CPU if needed.

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- HEIC (.heic) - iPhone photos
- WebP (.webp)

## Tips for Best Results

1. **Photography**:
   - Use good lighting (no harsh shadows)
   - Plain background (will be removed automatically)
   - Center the object in frame
   - Capture full object from one side

2. **Processing**:
   - Higher resolution = better detail but slower
   - Simpler objects work better than complex ones
   - Objects with clear depth variation work best

3. **Printing**:
   - Check `is_watertight` in metadata before printing
   - Review wall thickness warnings
   - Test print small scale first

## Batch Processing

The batch processor creates a comprehensive CSV report:

```bash
python image_to_3d.py --batch
```

This generates:
- Individual STL files for each image
- Batch report CSV with all metrics
- Summary of total material costs and retail value

## Troubleshooting

### Out of Memory Errors
- Reduce `target_resolution` in config.yaml
- Close other applications
- Enable CPU fallback

### Low Quality Output
- Increase `target_resolution` to 1024
- Increase `simplify_faces` to 50000
- Use better source images

### Non-Watertight Meshes
- Enable `ensure_watertight` in config
- Increase `poisson_depth` (9-10)
- Simplify source image background

## Performance

Typical processing times (GPU: NVIDIA RTX 3060):
- Preprocessing: 5-10 seconds
- Depth estimation: 10-20 seconds
- Mesh generation: 15-30 seconds
- Export & previews: 5-10 seconds

**Total: ~40-70 seconds per image**

CPU processing is approximately 3-5x slower.

## Business Use

This pipeline is designed for commercial 3D printing operations:

- **SKU Generation**: Automatic unique identifiers
- **Cost Calculation**: Material usage tracking
- **Pricing Suggestions**: Markup-based retail pricing
- **Print Time Estimates**: Production planning
- **Batch Reports**: Business analytics

## License

Commercial use permitted. See LICENSE file for details.

## Support

For issues and questions, check:
- Configuration file comments
- Log files in `logs/pipeline.log`
- Error messages in console output

## Changelog

### Version 1.0.0 (2025-11-17)
- Initial release
- MiDaS DPT-Large integration
- Poisson surface reconstruction
- Business analytics module
- Batch processing support
