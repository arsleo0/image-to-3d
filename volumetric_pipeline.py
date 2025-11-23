#!/usr/bin/env python3
"""
Image-to-3D Pipeline - Volumetric Version
Creates a 3D model with front and back depth (not just relief)
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import trimesh
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
import cv2
from scipy.ndimage import uniform_filter, gaussian_filter

def create_directories():
    """Create all required directories"""
    dirs = ['input', 'output/stl', 'output/previews', 'output/reports', 'processing', 'logs']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("Directories created.")

def load_image(image_path):
    """Load and prepare image"""
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize to 512x512
    img = img.resize((512, 512), Image.Resampling.LANCZOS)
    print(f"Image loaded: {image_path}")
    return img

def estimate_depth(image, device):
    """Estimate depth using MiDaS"""
    print("Loading depth model...")
    processor = DPTImageProcessor.from_pretrained('Intel/dpt-large')
    model = DPTForDepthEstimation.from_pretrained('Intel/dpt-large')
    model.to(device)
    model.eval()

    print("Estimating depth...")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Normalize
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    if depth_max - depth_min > 0:
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)

    # Smooth
    depth_map = gaussian_filter(depth_map, sigma=1)

    print(f"Depth map generated: {depth_map.shape}")
    return depth_map

def create_volumetric_mesh(depth_map, thickness_mode='mirror'):
    """
    Create volumetric 3D mesh from depth map

    thickness_mode options:
    - 'mirror': Mirror depth on back (best for figurines)
    - 'uniform': Uniform thickness shell
    - 'proportional': Back depth proportional to front
    """
    print(f"Creating volumetric mesh (mode: {thickness_mode})...")

    height, width = depth_map.shape

    # Parameters
    depth_scale = 60  # Height of the relief
    base_thickness = 5  # Minimum thickness
    step = 2  # Sampling step (1=detailed, 2=fast)

    # Create coordinate grids
    y_coords = np.arange(0, height, step)
    x_coords = np.arange(0, width, step)
    h_sampled = len(y_coords)
    w_sampled = len(x_coords)

    # Sample depth map
    depth_sampled = depth_map[::step, ::step]

    # Create front surface vertices
    xx, yy = np.meshgrid(x_coords, y_coords)
    zz_front = depth_sampled * depth_scale + base_thickness

    front_vertices = np.stack([
        xx.flatten(),
        yy.flatten(),
        zz_front.flatten()
    ], axis=1)

    # Create back surface vertices based on mode
    if thickness_mode == 'mirror':
        # Mirror the depth (creates double-sided relief)
        zz_back = -depth_sampled * depth_scale * 0.5
    elif thickness_mode == 'uniform':
        # Uniform thickness
        zz_back = np.zeros_like(depth_sampled)
    else:  # proportional
        # Back depth proportional to front
        zz_back = -depth_sampled * depth_scale * 0.3

    back_vertices = np.stack([
        xx.flatten(),
        yy.flatten(),
        zz_back.flatten()
    ], axis=1)

    # Combine vertices
    n_front = len(front_vertices)
    all_vertices = np.vstack([front_vertices, back_vertices])

    # Create faces for front surface
    front_faces = []
    for i in range(h_sampled - 1):
        for j in range(w_sampled - 1):
            v0 = i * w_sampled + j
            v1 = v0 + 1
            v2 = v0 + w_sampled + 1
            v3 = v0 + w_sampled

            # Two triangles per quad
            front_faces.append([v0, v1, v2])
            front_faces.append([v0, v2, v3])

    # Create faces for back surface (reversed winding)
    back_faces = []
    for i in range(h_sampled - 1):
        for j in range(w_sampled - 1):
            v0 = n_front + i * w_sampled + j
            v1 = v0 + 1
            v2 = v0 + w_sampled + 1
            v3 = v0 + w_sampled

            # Reversed winding for back face
            back_faces.append([v0, v2, v1])
            back_faces.append([v0, v3, v2])

    # Create side faces to close the mesh
    side_faces = []

    # Top edge (y = 0)
    for j in range(w_sampled - 1):
        v_front_0 = j
        v_front_1 = j + 1
        v_back_0 = n_front + j
        v_back_1 = n_front + j + 1

        side_faces.append([v_front_0, v_back_0, v_back_1])
        side_faces.append([v_front_0, v_back_1, v_front_1])

    # Bottom edge (y = max)
    for j in range(w_sampled - 1):
        v_front_0 = (h_sampled - 1) * w_sampled + j
        v_front_1 = v_front_0 + 1
        v_back_0 = n_front + (h_sampled - 1) * w_sampled + j
        v_back_1 = v_back_0 + 1

        side_faces.append([v_front_0, v_front_1, v_back_1])
        side_faces.append([v_front_0, v_back_1, v_back_0])

    # Left edge (x = 0)
    for i in range(h_sampled - 1):
        v_front_0 = i * w_sampled
        v_front_1 = (i + 1) * w_sampled
        v_back_0 = n_front + i * w_sampled
        v_back_1 = n_front + (i + 1) * w_sampled

        side_faces.append([v_front_0, v_front_1, v_back_1])
        side_faces.append([v_front_0, v_back_1, v_back_0])

    # Right edge (x = max)
    for i in range(h_sampled - 1):
        v_front_0 = i * w_sampled + (w_sampled - 1)
        v_front_1 = (i + 1) * w_sampled + (w_sampled - 1)
        v_back_0 = n_front + i * w_sampled + (w_sampled - 1)
        v_back_1 = n_front + (i + 1) * w_sampled + (w_sampled - 1)

        side_faces.append([v_front_0, v_back_0, v_back_1])
        side_faces.append([v_front_0, v_back_1, v_front_1])

    # Combine all faces
    all_faces = np.array(front_faces + back_faces + side_faces)

    # Create mesh
    mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)

    print(f"Volumetric mesh created: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh

def optimize_mesh(mesh):
    """Optimize mesh for 3D printing"""
    print("Optimizing mesh...")

    # Merge close vertices
    mesh.merge_vertices()

    # Remove degenerate faces
    mesh.update_faces(mesh.nondegenerate_faces())

    # Remove duplicate faces
    mesh.update_faces(mesh.unique_faces())

    # Fix normals
    trimesh.repair.fix_normals(mesh)

    # Fill holes
    trimesh.repair.fill_holes(mesh)

    # Scale to 100mm max dimension
    max_dim = max(mesh.extents)
    if max_dim > 0:
        scale = 100.0 / max_dim
        mesh.apply_scale(scale)

    # Center the mesh
    mesh.vertices -= mesh.centroid

    # Place on build plate
    mesh.vertices[:, 2] -= mesh.bounds[0, 2]

    print(f"Optimized: {mesh.extents[0]:.1f} x {mesh.extents[1]:.1f} x {mesh.extents[2]:.1f} mm")
    print(f"Watertight: {mesh.is_watertight}")

    return mesh

def calculate_metrics(mesh):
    """Calculate business metrics"""
    if mesh.is_watertight:
        volume = abs(mesh.volume)
    else:
        # Estimate volume for non-watertight mesh
        volume = abs(mesh.volume) * 0.9

    weight = (volume / 1000.0) * 1.24  # PLA density
    cost = weight * 0.02
    price = cost * 5.0

    sku = f"3DP-VOL-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    metrics = {
        'sku': sku,
        'volume_mm3': round(volume, 2),
        'weight_g': round(weight, 2),
        'cost_eur': round(cost, 2),
        'price_eur': round(price, 2),
        'dimensions': {
            'x': round(mesh.extents[0], 2),
            'y': round(mesh.extents[1], 2),
            'z': round(mesh.extents[2], 2)
        },
        'vertices': len(mesh.vertices),
        'faces': len(mesh.faces),
        'watertight': mesh.is_watertight
    }

    print(f"\nSKU: {sku}")
    print(f"Volume: {metrics['volume_mm3']} mm3")
    print(f"Weight: {metrics['weight_g']} g")
    print(f"Cost: EUR {metrics['cost_eur']}")
    print(f"Price: EUR {metrics['price_eur']}")

    return metrics

def export_stl(mesh, output_path):
    """Export mesh as STL"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mesh.export(str(output_path), file_type='stl')

    if output_path.exists() and output_path.stat().st_size > 0:
        size_mb = output_path.stat().st_size / 1e6
        print(f"\nSTL saved: {output_path} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"\nERROR: Failed to save STL")
        return False

def save_metadata(metrics, output_path):
    """Save metadata as JSON"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metadata saved: {output_path}")

def main():
    """Main function"""
    print("=" * 50)
    print("VOLUMETRIC IMAGE-TO-3D PIPELINE")
    print("Creates 3D models with depth on both sides")
    print("=" * 50)

    create_directories()

    # Parse arguments
    if len(sys.argv) < 2:
        print("\nUsage: python volumetric_pipeline.py <image_path> [mode]")
        print("\nModes:")
        print("  mirror      - Mirror depth on back (default, best for figurines)")
        print("  uniform     - Uniform thickness shell")
        print("  proportional - Back depth proportional to front")
        print("\nExample: python volumetric_pipeline.py input/photo.jpg mirror")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    thickness_mode = sys.argv[2] if len(sys.argv) > 2 else 'mirror'

    if not input_path.exists():
        print(f"\nError: File not found: {input_path}")
        sys.exit(1)

    # Detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("\nUsing CPU (slower)")

    try:
        # 1. Load image
        image = load_image(input_path)

        # 2. Estimate depth
        depth_map = estimate_depth(image, device)

        # Save depth
        depth_path = Path('processing') / f"{input_path.stem}_depth.png"
        cv2.imwrite(str(depth_path), (depth_map * 255).astype(np.uint8))
        print(f"Depth saved: {depth_path}")

        # 3. Create volumetric mesh
        mesh = create_volumetric_mesh(depth_map, thickness_mode)

        # 4. Optimize
        mesh = optimize_mesh(mesh)

        # 5. Calculate metrics
        metrics = calculate_metrics(mesh)

        # 6. Export
        stl_path = Path('output/stl') / f"{metrics['sku']}.stl"
        if not export_stl(mesh, stl_path):
            sys.exit(1)

        # 7. Save metadata
        json_path = Path('output/reports') / f"{metrics['sku']}.json"
        save_metadata(metrics, json_path)

        print("\n" + "=" * 50)
        print("SUCCESS!")
        print(f"STL: {stl_path}")
        print("=" * 50)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
