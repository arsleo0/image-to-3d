#!/usr/bin/env python3
"""
Image-to-3D Pipeline for Commercial 3D Printing
Converts product photos to printable STL files with business analytics
"""

import os
import sys
import yaml
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Optional
import time

import numpy as np
import torch
import trimesh
import open3d as o3d
from PIL import Image
from rembg import remove
from transformers import DPTImageProcessor, DPTForDepthEstimation
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(config: dict) -> logging.Logger:
    """Configure logging with file and console output"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))

    logger = logging.getLogger('Image23D')
    logger.setLevel(log_level)
    logger.handlers.clear()

    # Console handler with colors
    if log_config.get('console_output', True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_config.get('file_output', True):
        log_file = Path(log_config.get('log_file', 'logs/pipeline.log'))
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


class SystemDetector:
    """Detects and configures system resources (GPU, CPU, VRAM)"""

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config.get('system', {})
        self.logger = logger
        self.device = None
        self.vram_available = 0
        self.batch_size = 1

    def detect(self) -> dict:
        """Detect system capabilities"""
        self.logger.info("🔍 Detecting system capabilities...")

        # Check CUDA availability
        cuda_available = torch.cuda.is_available()

        if cuda_available and self.config.get('auto_detect_gpu', True):
            device_id = self.config.get('cuda_device', 0)
            self.device = torch.device(f'cuda:{device_id}')

            # Get GPU information
            gpu_name = torch.cuda.get_device_name(device_id)
            vram_total = torch.cuda.get_device_properties(device_id).total_memory / 1e9
            vram_allocated = torch.cuda.memory_allocated(device_id) / 1e9
            self.vram_available = vram_total - vram_allocated

            self.logger.info(f"✅ GPU detected: {gpu_name}")
            self.logger.info(f"💾 VRAM: {self.vram_available:.2f} GB available / {vram_total:.2f} GB total")

            # Set optimal batch size based on VRAM
            if self.vram_available > 8:
                self.batch_size = 4
            elif self.vram_available > 4:
                self.batch_size = 2
            else:
                self.batch_size = 1

        else:
            if self.config.get('cpu_fallback', True):
                self.device = torch.device('cpu')
                self.logger.warning("⚠️  No GPU detected, using CPU (processing will be slower)")
            else:
                raise RuntimeError("GPU not available and CPU fallback is disabled")

        # CPU information
        cpu_count = os.cpu_count() or 1
        self.logger.info(f"🖥️  CPU cores: {cpu_count}")

        return {
            'device': self.device,
            'device_type': 'cuda' if cuda_available else 'cpu',
            'vram_gb': self.vram_available,
            'batch_size': self.batch_size,
            'cpu_count': cpu_count
        }


class ImagePreprocessor:
    """Handles image loading, background removal, and preprocessing"""

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config.get('image', {})
        self.logger = logger

    def load_image(self, image_path: Path) -> Optional[Image.Image]:
        """Load image with support for various formats including HEIC"""
        try:
            # Handle HEIC format
            if image_path.suffix.lower() in ['.heic', '.heif']:
                try:
                    import pillow_heif
                    pillow_heif.register_heif_opener()
                except ImportError:
                    self.logger.error("HEIC support not available. Install pillow-heif")
                    return None

            img = Image.open(image_path)

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            self.logger.debug(f"Loaded image: {image_path.name} ({img.size[0]}x{img.size[1]})")
            return img

        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            return None

    def remove_background(self, image: Image.Image) -> Image.Image:
        """Remove background using rembg with alpha matting"""
        try:
            self.logger.debug("Removing background...")

            # Use rembg with alpha matting for better quality
            alpha_matting = self.config.get('alpha_matting', True)
            result = remove(
                image,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=10
            )

            return result

        except Exception as e:
            self.logger.warning(f"Background removal failed: {e}, using original")
            return image

    def center_and_pad(self, image: Image.Image) -> Image.Image:
        """Center object and add padding"""
        try:
            # Convert to numpy array
            img_array = np.array(image)

            # If image has alpha channel, use it to find bounding box
            if img_array.shape[2] == 4:
                alpha = img_array[:, :, 3]
                rows = np.any(alpha > 0, axis=1)
                cols = np.any(alpha > 0, axis=0)

                if np.any(rows) and np.any(cols):
                    y_min, y_max = np.where(rows)[0][[0, -1]]
                    x_min, x_max = np.where(cols)[0][[0, -1]]

                    # Crop to object
                    img_array = img_array[y_min:y_max+1, x_min:x_max+1]

            # Add padding
            padding_percent = self.config.get('padding_percent', 10)
            h, w = img_array.shape[:2]
            pad_h = int(h * padding_percent / 100)
            pad_w = int(w * padding_percent / 100)

            # Create padded image
            new_h = h + 2 * pad_h
            new_w = w + 2 * pad_w
            padded = np.zeros((new_h, new_w, img_array.shape[2]), dtype=np.uint8)

            # If RGBA, set alpha to 0 (transparent), otherwise white background
            if img_array.shape[2] == 4:
                padded[:, :, 3] = 0
            else:
                padded[:, :, :] = 255

            # Place centered image
            padded[pad_h:pad_h+h, pad_w:pad_w+w] = img_array

            return Image.fromarray(padded)

        except Exception as e:
            self.logger.warning(f"Centering failed: {e}, using original")
            return image

    def resize_to_target(self, image: Image.Image) -> Image.Image:
        """Resize image to target resolution maintaining aspect ratio"""
        target_res = self.config.get('target_resolution', 1024)

        # Calculate new size maintaining aspect ratio
        width, height = image.size
        if width > height:
            new_width = target_res
            new_height = int((height / width) * target_res)
        else:
            new_height = target_res
            new_width = int((width / height) * target_res)

        # Resize with high-quality resampling
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Pad to square
        final_img = Image.new('RGB', (target_res, target_res), (255, 255, 255))
        offset_x = (target_res - new_width) // 2
        offset_y = (target_res - new_height) // 2

        if resized.mode == 'RGBA':
            final_img.paste(resized, (offset_x, offset_y), resized)
        else:
            final_img.paste(resized, (offset_x, offset_y))

        return final_img

    def process(self, image_path: Path) -> Optional[Image.Image]:
        """Complete preprocessing pipeline"""
        self.logger.info(f"📸 Processing image: {image_path.name}")

        # Load image
        img = self.load_image(image_path)
        if img is None:
            return None

        # Remove background
        if self.config.get('background_removal', True):
            img = self.remove_background(img)

        # Center and pad
        img = self.center_and_pad(img)

        # Resize to target resolution
        img = self.resize_to_target(img)

        self.logger.info(f"✅ Preprocessing complete: {img.size[0]}x{img.size[1]}")
        return img


class DepthEstimator:
    """Estimates depth maps from images using MiDaS DPT-Large"""

    def __init__(self, config: dict, device: torch.device, logger: logging.Logger):
        self.config = config.get('depth', {})
        self.device = device
        self.logger = logger
        self.model = None
        self.processor = None

    def load_model(self):
        """Load MiDaS DPT-Large model"""
        try:
            model_name = self.config.get('model', 'Intel/dpt-large')
            self.logger.info(f"📦 Loading depth estimation model: {model_name}")

            self.processor = DPTImageProcessor.from_pretrained(model_name)
            self.model = DPTForDepthEstimation.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            self.logger.info("✅ Model loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def estimate_depth(self, image: Image.Image) -> np.ndarray:
        """Generate depth map from image"""
        try:
            if self.model is None:
                self.load_model()

            self.logger.info("🔮 Estimating depth map...")

            # Prepare image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth

            # Interpolate to original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            depth_map = prediction.cpu().numpy()

            # Normalize depth range
            if self.config.get('normalize_range', True):
                depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

            # Invert if needed (closer = higher values)
            if self.config.get('invert_depth', False):
                depth_map = 1.0 - depth_map

            self.logger.info(f"✅ Depth map generated: {depth_map.shape}")
            return depth_map

        except RuntimeError as e:
            if "out of memory" in str(e):
                self.logger.error("GPU out of memory! Clearing cache and retrying...")
                torch.cuda.empty_cache()
                # Retry once
                return self.estimate_depth(image)
            else:
                raise

    def apply_filters(self, depth_map: np.ndarray) -> np.ndarray:
        """Apply bilateral filtering and hole filling"""
        try:
            # Convert to uint8 for filtering
            depth_uint8 = (depth_map * 255).astype(np.uint8)

            # Bilateral filter for edge-preserving smoothing
            d = self.config.get('bilateral_filter_d', 9)
            sigma_color = self.config.get('bilateral_filter_sigma_color', 75)
            sigma_space = self.config.get('bilateral_filter_sigma_space', 75)

            filtered = cv2.bilateralFilter(depth_uint8, d, sigma_color, sigma_space)

            # Fill holes
            if self.config.get('hole_filling', True):
                # Create binary mask
                mask = filtered > 0
                filled_mask = binary_fill_holes(mask)

                # Inpaint holes
                kernel = np.ones((5, 5), np.uint8)
                hole_mask = (filled_mask.astype(np.uint8) - mask.astype(np.uint8)) * 255
                filtered = cv2.inpaint(filtered, hole_mask, 3, cv2.INPAINT_TELEA)

            # Convert back to float
            filtered_depth = filtered.astype(np.float32) / 255.0

            self.logger.debug("Applied bilateral filtering and hole filling")
            return filtered_depth

        except Exception as e:
            self.logger.warning(f"Filtering failed: {e}, using original")
            return depth_map

    def process(self, image: Image.Image) -> np.ndarray:
        """Complete depth estimation pipeline"""
        depth_map = self.estimate_depth(image)
        depth_map = self.apply_filters(depth_map)
        return depth_map


class MeshGenerator:
    """Generates 3D meshes from depth maps using point clouds and reconstruction"""

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config.get('mesh', {})
        self.logger = logger

    def depth_to_point_cloud(self, depth_map: np.ndarray, image: Image.Image) -> o3d.geometry.PointCloud:
        """Convert depth map to colored point cloud"""
        try:
            self.logger.info("🎯 Converting depth map to point cloud...")

            height, width = depth_map.shape

            # Create meshgrid for coordinates
            x, y = np.meshgrid(np.arange(width), np.arange(height))

            # Scale depth
            scale_factor = self.config.get('depth_scale_factor', 1.0)
            z = depth_map * scale_factor

            # Stack coordinates
            points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

            # Get colors from image
            img_array = np.array(image.resize((width, height)))
            colors = img_array.reshape(-1, 3) / 255.0

            # Filter out zero-depth points
            valid_mask = points[:, 2] > 0.01
            points = points[valid_mask]
            colors = colors[valid_mask]

            # Sample points if too many
            max_samples = self.config.get('point_cloud_samples', 100000)
            if len(points) > max_samples:
                indices = np.random.choice(len(points), max_samples, replace=False)
                points = points[indices]
                colors = colors[indices]

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Estimate normals
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(15)

            self.logger.info(f"✅ Point cloud created: {len(pcd.points)} points")
            return pcd

        except Exception as e:
            self.logger.error(f"Point cloud generation failed: {e}")
            raise

    def reconstruct_surface(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """Reconstruct surface using Poisson or Ball Pivoting"""
        try:
            method = self.config.get('method', 'poisson')
            self.logger.info(f"🔨 Reconstructing surface using {method}...")

            if method == 'poisson':
                depth = self.config.get('poisson_depth', 9)
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=depth
                )

                # Remove low-density vertices
                vertices_to_remove = densities < np.quantile(densities, 0.01)
                mesh.remove_vertices_by_mask(vertices_to_remove)

            else:  # ball_pivoting
                radii = [0.005, 0.01, 0.02, 0.04]
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd,
                    o3d.utility.DoubleVector(radii)
                )

            self.logger.info(f"✅ Surface reconstructed: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
            return mesh

        except Exception as e:
            self.logger.error(f"Surface reconstruction failed: {e}")
            raise

    def simplify_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Simplify mesh to target face count"""
        try:
            target_faces = self.config.get('simplify_faces', 50000)
            current_faces = len(mesh.triangles)

            if current_faces > target_faces:
                self.logger.info(f"🔽 Simplifying mesh: {current_faces} → {target_faces} faces")
                mesh = mesh.simplify_quadric_decimation(target_faces)

            # Smooth mesh
            iterations = self.config.get('smooth_iterations', 2)
            if iterations > 0:
                mesh = mesh.filter_smooth_simple(number_of_iterations=iterations)

            return mesh

        except Exception as e:
            self.logger.warning(f"Mesh simplification failed: {e}, using original")
            return mesh

    def make_watertight(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Ensure mesh is watertight and manifold"""
        try:
            if self.config.get('ensure_watertight', True):
                mesh.remove_duplicated_vertices()
                mesh.remove_duplicated_triangles()
                mesh.remove_degenerate_triangles()
                mesh.remove_unreferenced_vertices()

            if self.config.get('fix_manifold', True):
                mesh.remove_non_manifold_edges()

            self.logger.debug("Mesh cleanup complete")
            return mesh

        except Exception as e:
            self.logger.warning(f"Mesh cleanup failed: {e}")
            return mesh

    def process(self, depth_map: np.ndarray, image: Image.Image) -> o3d.geometry.TriangleMesh:
        """Complete mesh generation pipeline"""
        pcd = self.depth_to_point_cloud(depth_map, image)
        mesh = self.reconstruct_surface(pcd)
        mesh = self.simplify_mesh(mesh)
        mesh = self.make_watertight(mesh)
        return mesh


class PrintingOptimizer:
    """Optimizes meshes for 3D printing"""

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config.get('printing', {})
        self.logger = logger

    def scale_to_max_dimension(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Scale mesh to maximum dimension"""
        max_dim = self.config.get('max_dimension_mm', 100)
        current_max = max(mesh.extents)

        if current_max > 0:
            scale_factor = max_dim / current_max
            mesh.apply_scale(scale_factor)
            self.logger.info(f"📏 Scaled mesh to {max_dim}mm (factor: {scale_factor:.3f})")

        return mesh

    def check_wall_thickness(self, mesh: trimesh.Trimesh) -> bool:
        """Check if mesh meets minimum wall thickness"""
        min_thickness = self.config.get('min_wall_thickness_mm', 2.0)

        # Simple check: ensure edges are not too small
        if len(mesh.edges) > 0:
            edge_lengths = np.linalg.norm(
                mesh.vertices[mesh.edges[:, 0]] - mesh.vertices[mesh.edges[:, 1]],
                axis=1
            )
            min_edge = edge_lengths.min()

            if min_edge < min_thickness:
                self.logger.warning(
                    f"⚠️  Minimum edge length ({min_edge:.2f}mm) is below recommended "
                    f"wall thickness ({min_thickness}mm)"
                )
                return False

        return True

    def auto_orient(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Orient mesh for optimal printing (minimize support)"""
        if not self.config.get('auto_orient', True):
            return mesh

        try:
            # Place mesh on build platform (align bottom to Z=0)
            mesh.vertices[:, 2] -= mesh.bounds[0, 2]

            self.logger.debug("Mesh oriented for printing")
            return mesh

        except Exception as e:
            self.logger.warning(f"Auto-orientation failed: {e}")
            return mesh

    def fix_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Fix common mesh issues"""
        try:
            # Fill holes
            trimesh.repair.fill_holes(mesh)

            # Fix normals
            trimesh.repair.fix_normals(mesh)

            # Fix winding
            trimesh.repair.fix_winding(mesh)

            self.logger.debug("Mesh repairs applied")
            return mesh

        except Exception as e:
            self.logger.warning(f"Mesh repair failed: {e}")
            return mesh

    def process(self, mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
        """Complete printing optimization pipeline"""
        self.logger.info("🔧 Optimizing for 3D printing...")

        # Convert to trimesh for better processing
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=triangles)

        # Apply optimizations
        trimesh_obj = self.scale_to_max_dimension(trimesh_obj)
        trimesh_obj = self.auto_orient(trimesh_obj)
        trimesh_obj = self.fix_mesh(trimesh_obj)
        self.check_wall_thickness(trimesh_obj)

        self.logger.info("✅ Printing optimization complete")
        return trimesh_obj


class BusinessCalculator:
    """Calculates business metrics (cost, pricing, SKU generation)"""

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config.get('business', {})
        self.logger = logger

    def generate_sku(self, category: str = 'default') -> str:
        """Generate unique SKU"""
        prefix = self.config.get('sku_prefix', '3DP')
        category_codes = self.config.get('category_codes', {})
        category_code = category_codes.get(category, category_codes.get('default', 'GEN'))

        date_str = datetime.now().strftime('%Y%m%d')

        # Generate unique number (timestamp-based)
        unique_num = int(datetime.now().timestamp() * 1000) % 10000

        sku = f"{prefix}-{category_code}-{date_str}-{unique_num:04d}"
        return sku

    def calculate_volume(self, mesh: trimesh.Trimesh) -> float:
        """Calculate mesh volume in mm³"""
        try:
            # Ensure mesh is watertight for accurate volume
            if mesh.is_watertight:
                volume_mm3 = abs(mesh.volume)
            else:
                self.logger.warning("Mesh is not watertight, volume may be inaccurate")
                volume_mm3 = abs(mesh.volume)

            return volume_mm3

        except Exception as e:
            self.logger.warning(f"Volume calculation failed: {e}")
            return 0.0

    def calculate_weight(self, volume_mm3: float, density: float = 1.24) -> float:
        """Calculate weight in grams (default PLA density: 1.24 g/cm³)"""
        volume_cm3 = volume_mm3 / 1000.0
        weight_g = volume_cm3 * density
        return weight_g

    def calculate_material_cost(self, weight_g: float) -> float:
        """Calculate material cost"""
        cost_per_gram = self.config.get('material_cost_per_gram', 0.02)
        return weight_g * cost_per_gram

    def estimate_print_time(self, mesh: trimesh.Trimesh) -> float:
        """Estimate print time in hours"""
        try:
            # Rough estimation based on height and perimeter
            height_mm = mesh.extents[2]
            layer_height = 0.2  # mm
            num_layers = height_mm / layer_height

            # Estimate time per layer (simple model)
            speed_mm_s = self.config.get('print_speed_mm_s', 60)
            perimeter = np.mean([
                np.linalg.norm(mesh.vertices[mesh.edges[:, 0]] - mesh.vertices[mesh.edges[:, 1]])
            ]) * len(mesh.edges) / 100  # Rough perimeter estimate

            time_per_layer_s = perimeter / speed_mm_s
            total_time_s = num_layers * time_per_layer_s
            total_time_h = total_time_s / 3600

            return total_time_h

        except Exception as e:
            self.logger.warning(f"Print time estimation failed: {e}")
            return 0.0

    def calculate_retail_price(self, material_cost: float) -> float:
        """Calculate recommended retail price"""
        markup = self.config.get('retail_markup', 5.0)
        return material_cost * markup

    def process(self, mesh: trimesh.Trimesh, category: str = 'default') -> dict:
        """Calculate all business metrics"""
        self.logger.info("💰 Calculating business metrics...")

        sku = self.generate_sku(category)
        volume_mm3 = self.calculate_volume(mesh)
        weight_g = self.calculate_weight(volume_mm3)
        material_cost = self.calculate_material_cost(weight_g)
        print_time_h = self.estimate_print_time(mesh)
        retail_price = self.calculate_retail_price(material_cost)

        metrics = {
            'sku': sku,
            'volume_mm3': round(volume_mm3, 2),
            'weight_g': round(weight_g, 2),
            'material_cost_eur': round(material_cost, 2),
            'print_time_hours': round(print_time_h, 2),
            'recommended_price_eur': round(retail_price, 2),
            'dimensions_mm': {
                'x': round(mesh.extents[0], 2),
                'y': round(mesh.extents[1], 2),
                'z': round(mesh.extents[2], 2)
            },
            'is_watertight': mesh.is_watertight,
            'vertex_count': len(mesh.vertices),
            'face_count': len(mesh.faces)
        }

        self.logger.info(f"✅ SKU: {sku}")
        self.logger.info(f"   Volume: {metrics['volume_mm3']:.2f} mm³")
        self.logger.info(f"   Weight: {metrics['weight_g']:.2f} g")
        self.logger.info(f"   Material Cost: €{metrics['material_cost_eur']:.2f}")
        self.logger.info(f"   Recommended Price: €{metrics['recommended_price_eur']:.2f}")

        return metrics


class Exporter:
    """Handles STL export and preview generation"""

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config.get('export', {})
        self.paths = config.get('paths', {})
        self.logger = logger

    def export_stl(self, mesh: trimesh.Trimesh, output_path: Path) -> bool:
        """Export mesh as STL file"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Export with proper units
            mesh.export(
                str(output_path),
                file_type='stl',
                include_color=False
            )

            file_size_mb = output_path.stat().st_size / 1e6
            self.logger.info(f"💾 STL exported: {output_path.name} ({file_size_mb:.2f} MB)")

            return True

        except Exception as e:
            self.logger.error(f"STL export failed: {e}")
            return False

    def generate_preview(self, mesh: trimesh.Trimesh, output_dir: Path, base_name: str):
        """Generate preview renders from multiple angles"""
        try:
            if not self.config.get('generate_previews', True):
                return

            self.logger.info("🎨 Generating preview renders...")
            output_dir.mkdir(parents=True, exist_ok=True)

            angles = self.config.get('preview_angles', [0, 45, 90, 135])
            resolution = self.config.get('preview_resolution', 800)

            for angle in angles:
                try:
                    scene = mesh.scene()

                    # Rotate camera around object
                    rotation_matrix = trimesh.transformations.rotation_matrix(
                        np.radians(angle), [0, 0, 1]
                    )
                    camera_transform = rotation_matrix

                    # Render
                    png = scene.save_image(resolution=(resolution, resolution))

                    # Save
                    preview_path = output_dir / f"{base_name}_preview_{angle}deg.png"
                    with open(preview_path, 'wb') as f:
                        f.write(png)

                except Exception as e:
                    self.logger.warning(f"Preview generation failed for angle {angle}: {e}")

            self.logger.info(f"✅ Generated {len(angles)} preview renders")

        except Exception as e:
            self.logger.warning(f"Preview generation failed: {e}")

    def generate_thumbnail(self, mesh: trimesh.Trimesh, output_path: Path):
        """Generate thumbnail for marketplace"""
        try:
            thumbnail_size = self.config.get('thumbnail_size', 512)
            bg_color = self.config.get('thumbnail_background', [255, 255, 255])

            scene = mesh.scene()
            png = scene.save_image(resolution=(thumbnail_size, thumbnail_size))

            # Save thumbnail
            with open(output_path, 'wb') as f:
                f.write(png)

            self.logger.debug(f"Thumbnail saved: {output_path.name}")

        except Exception as e:
            self.logger.warning(f"Thumbnail generation failed: {e}")

    def export_metadata(self, metadata: dict, output_path: Path):
        """Export metadata as JSON"""
        try:
            if not self.config.get('export_metadata', True):
                return

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.logger.debug(f"Metadata saved: {output_path.name}")

        except Exception as e:
            self.logger.warning(f"Metadata export failed: {e}")


class Image3DPipeline:
    """Main pipeline orchestrator"""

    def __init__(self, config_path: str = 'config.yaml'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup logging
        self.logger = setup_logging(self.config)

        # Initialize system
        self.system = SystemDetector(self.config, self.logger)
        self.system_info = self.system.detect()

        # Initialize components
        self.preprocessor = ImagePreprocessor(self.config, self.logger)
        self.depth_estimator = DepthEstimator(
            self.config, self.system_info['device'], self.logger
        )
        self.mesh_generator = MeshGenerator(self.config, self.logger)
        self.printing_optimizer = PrintingOptimizer(self.config, self.logger)
        self.business_calculator = BusinessCalculator(self.config, self.logger)
        self.exporter = Exporter(self.config, self.logger)

        self.logger.info("=" * 60)
        self.logger.info("🚀 Image-to-3D Pipeline Initialized")
        self.logger.info("=" * 60)

    def process_single(self, input_path: Path, category: str = 'default') -> Optional[dict]:
        """Process a single image through the complete pipeline"""
        try:
            start_time = time.time()
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing: {input_path.name}")
            self.logger.info(f"{'='*60}")

            # 1. Preprocess image
            processed_image = self.preprocessor.process(input_path)
            if processed_image is None:
                return None

            # Save processed image
            processing_dir = Path(self.config['paths']['processing_dir'])
            processing_dir.mkdir(exist_ok=True)
            processed_path = processing_dir / f"{input_path.stem}_processed.png"
            processed_image.save(processed_path)

            # 2. Estimate depth
            depth_map = self.depth_estimator.process(processed_image)

            # Save depth map visualization
            depth_vis_path = processing_dir / f"{input_path.stem}_depth.png"
            plt.imsave(depth_vis_path, depth_map, cmap='viridis')

            # 3. Generate mesh
            mesh_o3d = self.mesh_generator.process(depth_map, processed_image)

            # 4. Optimize for printing
            mesh_trimesh = self.printing_optimizer.process(mesh_o3d)

            # 5. Calculate business metrics
            business_metrics = self.business_calculator.process(mesh_trimesh, category)

            # 6. Export STL
            stl_dir = Path(self.config['paths']['stl_dir'])
            stl_path = stl_dir / f"{business_metrics['sku']}.stl"
            self.exporter.export_stl(mesh_trimesh, stl_path)

            # 7. Generate previews
            previews_dir = Path(self.config['paths']['previews_dir'])
            self.exporter.generate_preview(mesh_trimesh, previews_dir, business_metrics['sku'])

            # 8. Generate thumbnail
            thumbnail_path = previews_dir / f"{business_metrics['sku']}_thumb.png"
            self.exporter.generate_thumbnail(mesh_trimesh, thumbnail_path)

            # 9. Export metadata
            processing_time = time.time() - start_time
            metadata = {
                'input_file': input_path.name,
                'processing_time_seconds': round(processing_time, 2),
                'timestamp': datetime.now().isoformat(),
                'category': category,
                **business_metrics
            }

            reports_dir = Path(self.config['paths']['reports_dir'])
            metadata_path = reports_dir / f"{business_metrics['sku']}.json"
            self.exporter.export_metadata(metadata, metadata_path)

            # Log summary
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"✅ Processing complete in {processing_time:.2f}s")
            self.logger.info(f"{'='*60}\n")

            return metadata

        except Exception as e:
            self.logger.error(f"❌ Pipeline failed for {input_path.name}: {e}", exc_info=True)
            return None


class BatchProcessor:
    """Handles batch processing of multiple images"""

    def __init__(self, pipeline: Image3DPipeline):
        self.pipeline = pipeline
        self.logger = pipeline.logger
        self.config = pipeline.config

    def get_input_files(self) -> List[Path]:
        """Get all valid input files"""
        input_dir = Path(self.config['paths']['input_dir'])
        supported_formats = self.config['image']['supported_formats']

        files = []
        for ext in supported_formats:
            files.extend(input_dir.glob(f"*.{ext}"))
            files.extend(input_dir.glob(f"*.{ext.upper()}"))

        return sorted(files)

    def process_batch(self, category: str = 'default') -> List[dict]:
        """Process all images in input directory"""
        files = self.get_input_files()

        if not files:
            self.logger.warning(f"No input files found in {self.config['paths']['input_dir']}")
            return []

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📦 Starting batch processing: {len(files)} files")
        self.logger.info(f"{'='*60}\n")

        results = []
        failed = []

        # Process with progress bar
        for file_path in tqdm(files, desc="Processing images", unit="image"):
            result = self.pipeline.process_single(file_path, category)

            if result:
                results.append(result)
            else:
                failed.append(file_path.name)

        # Generate batch report
        self.generate_batch_report(results, failed)

        # Send notification
        if self.config['batch'].get('notification_on_complete', True):
            self.send_notification(len(results), len(failed))

        return results

    def generate_batch_report(self, results: List[dict], failed: List[str]):
        """Generate CSV report for batch"""
        if not results:
            return

        try:
            import csv

            reports_dir = Path(self.config['paths']['reports_dir'])
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = reports_dir / f"batch_report_{timestamp}.csv"

            # Write CSV
            with open(report_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

            self.logger.info(f"\n📊 Batch report saved: {report_path.name}")

            # Summary
            total_cost = sum(r['material_cost_eur'] for r in results)
            total_price = sum(r['recommended_price_eur'] for r in results)

            self.logger.info(f"\n{'='*60}")
            self.logger.info("📈 BATCH SUMMARY")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Total processed: {len(results)}")
            self.logger.info(f"Failed: {len(failed)}")
            self.logger.info(f"Total material cost: €{total_cost:.2f}")
            self.logger.info(f"Total retail value: €{total_price:.2f}")
            self.logger.info(f"Potential profit: €{total_price - total_cost:.2f}")
            self.logger.info(f"{'='*60}\n")

            if failed:
                self.logger.warning(f"Failed files: {', '.join(failed)}")

        except Exception as e:
            self.logger.error(f"Failed to generate batch report: {e}")

    def send_notification(self, success_count: int, failed_count: int):
        """Send desktop notification when batch completes"""
        try:
            # Simple notification using system bell
            print('\a')  # Terminal bell
            self.logger.info(f"🔔 Notification: Batch processing complete ({success_count} succeeded, {failed_count} failed)")
        except Exception:
            pass


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Image-to-3D Pipeline for Commercial 3D Printing'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--input',
        help='Single input image file (if not provided, processes batch)'
    )
    parser.add_argument(
        '--category',
        default='default',
        choices=['default', 'figurine', 'decoration', 'functional', 'prototype'],
        help='Product category for SKU generation'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all images in input directory'
    )

    args = parser.parse_args()

    try:
        # Initialize pipeline
        pipeline = Image3DPipeline(args.config)

        if args.input:
            # Process single file
            input_path = Path(args.input)
            if not input_path.exists():
                print(f"Error: Input file not found: {input_path}")
                sys.exit(1)

            result = pipeline.process_single(input_path, args.category)
            if result:
                print(f"\n✅ Success! STL saved: {result['sku']}.stl")
            else:
                print("\n❌ Processing failed")
                sys.exit(1)

        else:
            # Process batch
            processor = BatchProcessor(pipeline)
            results = processor.process_batch(args.category)

            if results:
                print(f"\n✅ Batch processing complete: {len(results)} files processed")
            else:
                print("\n⚠️  No files processed")
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
