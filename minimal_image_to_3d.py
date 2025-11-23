#!/usr/bin/env python3
"""
Minimal Image-to-3D Pipeline - Easy to copy version
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import trimesh
import open3d as o3d
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
import cv2

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

    # Resize to image size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Normalize
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

    print(f"Depth map generated: {depth_map.shape}")
    return depth_map

def depth_to_mesh(depth_map, image):
    """Convert depth map to 3D mesh"""
    print("Creating point cloud...")
    height, width = depth_map.shape

    # Create coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_map * 100  # Scale depth

    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Get colors
    img_array = np.array(image.resize((width, height)))
    colors = img_array.reshape(-1, 3) / 255.0

    # Filter valid points
    valid = points[:, 2] > 1
    points = points[valid]
    colors = colors[valid]

    # Sample if too many
    if len(points) > 50000:
        idx = np.random.choice(len(points), 50000, replace=False)
        points = points[idx]
        colors = colors[idx]

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(15)

    print(f"Point cloud: {len(pcd.points)} points")

    # Reconstruct surface
    print("Reconstructing surface (Poisson)...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

    # Clean mesh
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    print(f"Mesh created: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
    return mesh

def optimize_for_printing(mesh_o3d):
    """Optimize mesh for 3D printing"""
    print("Optimizing for printing...")

    # Convert to trimesh
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Scale to 100mm max
    max_dim = max(mesh.extents)
    if max_dim > 0:
        scale = 100.0 / max_dim
        mesh.apply_scale(scale)

    # Fix mesh
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh)

    # Orient for printing
    mesh.vertices[:, 2] -= mesh.bounds[0, 2]

    print(f"Optimized: {mesh.extents[0]:.1f} x {mesh.extents[1]:.1f} x {mesh.extents[2]:.1f} mm")
    return mesh

def calculate_metrics(mesh):
    """Calculate business metrics"""
    volume = abs(mesh.volume) if mesh.is_watertight else abs(mesh.volume)
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

    return metrics

def export_stl(mesh, output_path):
    """Export mesh as STL"""
    # Ensure directory exists
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
    print("MINIMAL IMAGE-TO-3D PIPELINE")
    print("=" * 50)

    # Create directories
    create_directories()

    # Check arguments
    if len(sys.argv) < 2:
        print("\nUsage: python minimal_image_to_3d.py <image_path>")
        print("Example: python minimal_image_to_3d.py input/photo.jpg")
        sys.exit(1)

    input_path = Path(sys.argv[1])
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

        # Save depth visualization
        depth_path = Path('processing') / f"{input_path.stem}_depth.png"
        cv2.imwrite(str(depth_path), (depth_map * 255).astype(np.uint8))
        print(f"Depth saved: {depth_path}")

        # 3. Create mesh
        mesh_o3d = depth_to_mesh(depth_map, image)

        # 4. Optimize
        mesh = optimize_for_printing(mesh_o3d)

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
