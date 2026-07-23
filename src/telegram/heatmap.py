"""Heatmap image generation for Telegram bot."""
# mypy: ignore-errors

import io
from PIL import Image, ImageDraw, ImageFilter


def generate_heatmap_image(
    points: list,
    width: int = 640,
    height: int = 480,
    native_width: int = 1920,
    native_height: int = 1080,
) -> io.BytesIO:
    """
    Generate a heatmap image from detection points.

    Args:
        points: List of {center_x, center_y, weight} dicts
        width: Output image width
        height: Output image height
        native_width: Native camera resolution width
        native_height: Native camera resolution height

    Returns:
        BytesIO containing PNG image data
    """
    # Create base image
    img = Image.new("RGB", (width, height), (26, 29, 35))
    draw = ImageDraw.Draw(img)

    if not points:
        # No data - draw placeholder text
        draw.text((width // 2 - 80, height // 2 - 10), "No heatmap data", fill=(154, 160, 166))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf

    # Normalize weights
    max_weight = max(p.get("weight", 1) for p in points)
    if max_weight == 0:
        max_weight = 1

    # Create heatmap layer
    heatmap = Image.new("L", (width, height), 0)
    heat_draw = ImageDraw.Draw(heatmap)

    for p in points:
        x = (p["center_x"] / native_width) * width
        y = (p["center_y"] / native_height) * height
        weight = p.get("weight", 1) / max_weight

        # Draw gradient circle
        radius = int(15 + weight * 25)
        intensity = int(weight * 255)

        for r in range(radius, 0, -2):
            alpha = int(intensity * (r / radius) * 0.4)
            heat_draw.ellipse(
                [x - r, y - r, x + r, y + r],
                fill=alpha,
            )

    # Apply Gaussian blur for smooth heatmap
    heatmap = heatmap.filter(ImageFilter.GaussianBlur(radius=15))

    # Apply colormap (yellow-orange-red)
    colored = Image.new("RGB", (width, height))
    pixels = colored.load()
    if pixels is None:
        return img
    heat_pixels = heatmap.load()
    if heat_pixels is None:
        return img

    for y in range(height):
        for x in range(width):
            val = heat_pixels[x, y]
            if val > 0:
                # Yellow -> Orange -> Red
                if val < 128:
                    r = 255
                    g = int(255 - (val / 128) * 155)
                    b = 0
                else:
                    r = 255
                    g = int(100 - ((val - 128) / 127) * 100)
                    b = 0
                pixels[x, y] = (r, g, b)

    # Blend heatmap with base
    img = Image.blend(img, colored, 0.6)

    # Draw legend
    draw = ImageDraw.Draw(img)
    legend_width = 120
    legend_height = 12
    legend_x = width - legend_width - 15
    legend_y = height - 25

    for i in range(legend_width):
        val = i / legend_width
        if val < 0.5:
            r = 255
            g = int(255 - val * 2 * 155)
        else:
            r = 255
            g = int(100 - (val - 0.5) * 2 * 100)
        draw.line([(legend_x + i, legend_y), (legend_x + i, legend_y + legend_height)], fill=(r, g, 0))

    draw.text((legend_x, legend_y - 12), "Low", fill=(154, 160, 166))
    draw.text((legend_x + legend_width - 20, legend_y - 12), "High", fill=(154, 160, 166))

    # Save to BytesIO
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf
