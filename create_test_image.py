#!/usr/bin/env python3
"""Generate a simple test image for the pipeline"""

from PIL import Image, ImageDraw
import numpy as np

# Create a simple image with a sphere-like object
width, height = 800, 800
img = Image.new('RGB', (width, height), color='white')
draw = ImageDraw.Draw(img)

# Draw a gradient sphere
center_x, center_y = width // 2, height // 2
radius = 250

for r in range(radius, 0, -2):
    # Calculate color based on distance from center (gradient effect)
    intensity = int(255 * (r / radius))
    color = (0, 100 + int(155 * (1 - r/radius)), 255 - int(100 * (r/radius)))
    draw.ellipse(
        [center_x - r, center_y - r, center_x + r, center_y + r],
        fill=color,
        outline=color
    )

# Add some highlights for 3D effect
highlight_offset = 80
draw.ellipse(
    [center_x - highlight_offset - 50, center_y - highlight_offset - 50,
     center_x - highlight_offset + 50, center_y - highlight_offset + 50],
    fill=(200, 220, 255),
    outline=(200, 220, 255)
)

# Save
img.save('input/test_sphere.png')
print("Test image created: input/test_sphere.png")
