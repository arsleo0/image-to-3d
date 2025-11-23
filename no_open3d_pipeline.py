#!/usr/bin/env python3
"""
Image-to-3D Pipeline - No Open3D Version
Works with Python 3.14
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
from scipy.ndimage import uniform_filter

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
    print("Loading depth model (first time takes a while)...")
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

    # Resize to image size
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
    else:
        depth_map = np.zeros_like(depth_map)

    # Smooth depth map
    depth_map = uniform_filter(depth_map, size=3)

    print(f"Depth map generated: {depth_map.shape}")
    return depth_map

def depth_to_mesh(depth_map, image):
    """Convert depth map to 3D mesh using marching cubes approach"""
    print("Creating 3D mesh from depth map...")

    height, width = depth_map.shape

    # Create vertex grid
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)

    # Scale depth for better 3D effect
    zz = depth_map * 50  # Height scale

    # Create vertices
    vertices = []
    faces = []

    # Sample every 2nd pixel for performance
    step = 2

    vertex_map = {}
    vertex_idx = 0

    for i in range(0, height - step, step):
        for j in range(0, width - step, step):
            # Get 4 corners of quad
            v0 = (j, i, zz[i, j])
            v1 = (j + step, i, zz[i, j + step])
            v2 = (j + step, i + step, zz[i + step, j + step])
            v3 = (j, i + step, zz[i + step, j])

            # Add vertices
            idx0 = vertex_idx
            vertices.append(v0)
            vertex_idx += 1

            idx1 = vertex_idx
            vertices.append(v1)
            vertex_idx += 1

            idx2 = vertex_idx
            vertices.append(v2)
            vertex_idx += 1

            idx3 = vertex_idx
            vertices.append(v3)
            vertex_idx += 1

            # Create two triangles for quad
            faces.append([idx0, idx1, idx2])
            faces.append([idx0, idx2, idx3])

    vertices = np.array(vertices)
    faces = np.array(faces)

    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Add back face for solid model
    back_vertices = vertices.copy()
    back_vertices[:, 2] = 0  # Flat back

    all_vertices = np.vstack([vertices, back_vertices])

    # Back faces (reversed winding)
    back_faces = faces.copy() + len(vertices)
    back_faces = back_faces[:, ::-1]  # Reverse winding

    all_faces = np.vstack([faces, back_faces])

    # Create sides
    # This is simplified - just creates a basic solid

    mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)

    print(f"Mesh created: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh

def optimize_mesh(mesh):
    """Optimize mesh for 3D printing"""
    print("Optimizing mesh...")

    # Remove degenerate faces
    mesh.update_faces(mesh.nondegenerate_faces())

    # Remove duplicate faces
    mesh.update_faces(mesh.unique_faces())

    # Merge close vertices
    mesh.merge_vertices()

    # Fix normals
    trimesh.repair.fix_normals(mesh)

    # Fill holes
    trimesh.repair.fill_holes(mesh)

    # Scale to 100mm max dimension
    max_dim = max(mesh.extents)
    if max_dim > 0:
        scale = 100.0 / max_dim
        mesh.apply_scale(scale)

    # Place on build plate
    mesh.vertices[:, 2] -= mesh.bounds[0, 2]

    print(f"Optimized: {mesh.extents[0]:.1f} x {mesh.extents[1]:.1f} x {mesh.extents[2]:.1f} mm")
    return mesh

def calculate_metrics(mesh):
    """Calculate business metrics"""
    volume = abs(mesh.volume) if mesh.is_watertight else abs(mesh.volume) * 0.8
    weight = (volume / 1000.0) * 1.24  # PLA density
    cost = weight * 0.02
    price = cost * 5.0

    # Generate SKU
    sku = f"3DP-{datetime.now().strftime('%Y%m%d%H%M%S')}"

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
    print(f"Watertight: {metrics['watertight']}")

    return metrics

def export_stl(mesh, output_path):
    """Export mesh as STL"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export
    mesh.export(str(output_path), file_type='stl')

    # Verify
    if output_path.exists() and output_path.stat().st_size > 0:
        size_mb = output_path.stat().st_size / 1e6
        print(f"\nSTL saved: {output_path} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"\nERROR: Failed to save STL to {output_path}")
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
    print("IMAGE-TO-3D PIPELINE (No Open3D)")
    print("Python 3.14 Compatible")
    print("=" * 50)

    # Create directories
    create_directories()

    # Check arguments
    if len(sys.argv) < 2:
        print("\nUsage: python no_open3d_pipeline.py <image_path>")
        print("Example: python no_open3d_pipeline.py input/photo.jpg")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"\nError: File not found: {input_path}")
        sys.exit(1)

    # Detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\nUsing GPU: {gpu_name}")
    else:
        device = torch.device('cpu')
        print("\nUsing CPU (this will be slower)")

    try:
        # 1. Load image
        image = load_image(input_path)

        # 2. Estimate depth
        depth_map = estimate_depth(image, device)

        # Save depth visualization
        depth_path = Path('processing') / f"{input_path.stem}_depth.png"
        depth_vis = (depth_map * 255).astype(np.uint8)
        cv2.imwrite(str(depth_path), depth_vis)
        print(f"Depth saved: {depth_path}")

        # 3. Create mesh
        mesh = depth_to_mesh(depth_map, image)

        # 4. Optimize
        mesh = optimize_mesh(mesh)

        # 5. Calculate metrics
        metrics = calculate_metrics(mesh)

        # 6. Export STL
        stl_path = Path('output/stl') / f"{metrics['sku']}.stl"
        success = export_stl(mesh, stl_path)

        if not success:
            print("\nPipeline failed!")
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
