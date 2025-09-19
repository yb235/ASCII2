#!/usr/bin/env python3
"""
Simplified High-Quality ASCII Converter - No OpenCV Dependencies

Uses only PIL and numpy to achieve 4K-like ASCII art quality through:
- Ultra high resolution (300+ characters wide)
- Advanced character sets with Unicode blocks
- Floyd-Steinberg dithering
- Histogram equalization with PIL
- Multiple output formats (text, HTML, ANSI color)
"""

import os
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from typing import Tuple, Optional, List, Dict
import html


class SimplifiedHighQualityASCII:
    """
    High-quality ASCII converter using only PIL and numpy.
    Achieves 4K-like quality without external dependencies.
    """
    
    # Ultra-detailed character sets
    CHAR_SETS = {
        'ascii_ultra': " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
        'blocks': " ░▒▓█",
        'sub_pixel': " ▘▝▀▖▌▞▛▗▚▐▜▄▙▟█",  # 4 sub-pixels per character
        'braille_simple': " ⠁⠃⠇⠏⠟⠿⣿",  # Simplified braille progression
        'gradients': " ░░▒▒▓▓██",
        'detailed': " .:-=+*#%@",
        'ultra_fine': " `.'\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    }
    
    # ANSI color codes for colored ASCII
    ANSI_COLORS = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'reset': '\033[0m'
    }
    
    def __init__(self, char_set: str = 'ascii_ultra', width: int = 320, 
                 output_format: str = 'text', use_color: bool = False):
        """
        Initialize the simplified high-quality ASCII converter.
        
        Args:
            char_set: Character set to use
            width: Output width (300+ for 4K-like quality)
            output_format: 'text', 'html', or 'ansi'
            use_color: Whether to use color (for ANSI/HTML output)
        """
        self.chars = self.CHAR_SETS.get(char_set, self.CHAR_SETS['ascii_ultra'])
        self.width = width
        self.output_format = output_format
        self.use_color = use_color
        self.char_len = len(self.chars)
        
        # Optimized character aspect ratio
        self.char_aspect_ratio = 0.43
    
    def histogram_equalization_pil(self, image: Image.Image) -> Image.Image:
        """
        Perform histogram equalization using only PIL.
        
        Args:
            image: PIL Image in grayscale
            
        Returns:
            Histogram equalized image
        """
        # Get histogram
        histogram = image.histogram()
        
        # Calculate cumulative distribution
        cdf = []
        cumsum = 0
        for count in histogram:
            cumsum += count
            cdf.append(cumsum)
        
        # Normalize CDF
        total_pixels = image.width * image.height
        cdf_normalized = [int(255 * c / total_pixels) for c in cdf]
        
        # Create lookup table
        lut = []
        for i in range(256):
            lut.append(cdf_normalized[i])
        
        # Apply lookup table
        return image.point(lut)
    
    def advanced_preprocess_pil(self, image: Image.Image) -> Image.Image:
        """
        Advanced preprocessing using only PIL operations.
        
        Args:
            image: PIL Image object
            
        Returns:
            Processed image optimized for ASCII conversion
        """
        # Convert to grayscale with optimized weights
        if image.mode != 'L':
            # Use ImageOps for better grayscale conversion
            gray_img = ImageOps.grayscale(image)
        else:
            gray_img = image.copy()
        
        # Histogram equalization for better contrast distribution
        gray_img = self.histogram_equalization_pil(gray_img)
        
        # Gamma correction using point operation
        gamma = 1.2
        gamma_table = [int(((i / 255.0) ** (1.0 / gamma)) * 255) for i in range(256)]
        gray_img = gray_img.point(gamma_table)
        
        # Enhanced contrast
        enhancer = ImageEnhance.Contrast(gray_img)
        gray_img = enhancer.enhance(1.6)
        
        # Edge detection and enhancement
        edges = gray_img.filter(ImageFilter.FIND_EDGES)
        # Blend edges with original for detail preservation
        gray_img = Image.blend(gray_img, edges, 0.2)
        
        # Unsharp mask effect
        blurred = gray_img.filter(ImageFilter.GaussianBlur(0.8))
        # Create unsharp mask by subtracting blur from original
        unsharp_array = np.array(gray_img).astype(np.float32) - np.array(blurred).astype(np.float32) * 0.5
        unsharp_array = np.clip(unsharp_array + np.array(gray_img).astype(np.float32), 0, 255).astype(np.uint8)
        gray_img = Image.fromarray(unsharp_array)
        
        # Final sharpening
        gray_img = gray_img.filter(ImageFilter.SHARPEN)
        
        return gray_img
    
    def floyd_steinberg_dither(self, image_array: np.ndarray) -> np.ndarray:
        """
        Floyd-Steinberg dithering implementation with numpy.
        
        Args:
            image_array: 2D numpy array of grayscale values
            
        Returns:
            Dithered array
        """
        height, width = image_array.shape
        dithered = image_array.astype(np.float32)
        
        for y in range(height):
            for x in range(width):
                old_pixel = dithered[y, x]
                # Quantize to character levels
                new_pixel = round(old_pixel * (self.char_len - 1) / 255) * 255 / (self.char_len - 1)
                dithered[y, x] = new_pixel
                
                error = old_pixel - new_pixel
                
                # Distribute error to neighbors
                if x + 1 < width:
                    dithered[y, x + 1] += error * 7/16
                if y + 1 < height:
                    if x > 0:
                        dithered[y + 1, x - 1] += error * 3/16
                    dithered[y + 1, x] += error * 5/16
                    if x + 1 < width:
                        dithered[y + 1, x + 1] += error * 1/16
        
        return np.clip(dithered, 0, 255)
    
    def calculate_output_size(self, image: Image.Image) -> Tuple[int, int]:
        """Calculate optimal output dimensions."""
        orig_width, orig_height = image.size
        aspect_ratio = orig_height / orig_width
        adjusted_height = int(self.width * aspect_ratio * self.char_aspect_ratio)
        
        # Ensure reasonable minimum height
        if adjusted_height < 50:
            adjusted_height = 50
            
        return self.width, adjusted_height
    
    def pixel_to_char_advanced(self, pixel_value: int) -> str:
        """
        Advanced pixel to character mapping with perceptual correction.
        
        Args:
            pixel_value: Pixel intensity (0-255)
            
        Returns:
            ASCII character
        """
        # Perceptual mapping (humans perceive brightness non-linearly)
        normalized = pixel_value / 255.0
        perceptual = normalized ** 0.75  # Adjust for better visual perception
        
        char_index = int(perceptual * (self.char_len - 1))
        char_index = max(0, min(char_index, self.char_len - 1))
        
        return self.chars[char_index]
    
    def rgb_to_ansi_color(self, r: int, g: int, b: int) -> str:
        """Convert RGB to closest ANSI color code."""
        # Simplified color mapping to 16 ANSI colors
        colors = [
            (0, 0, 0, 'black'),
            (128, 0, 0, 'red'),
            (0, 128, 0, 'green'),
            (128, 128, 0, 'yellow'),
            (0, 0, 128, 'blue'),
            (128, 0, 128, 'magenta'),
            (0, 128, 128, 'cyan'),
            (192, 192, 192, 'white'),
            (128, 128, 128, 'bright_black'),
            (255, 0, 0, 'bright_red'),
            (0, 255, 0, 'bright_green'),
            (255, 255, 0, 'bright_yellow'),
            (0, 0, 255, 'bright_blue'),
            (255, 0, 255, 'bright_magenta'),
            (0, 255, 255, 'bright_cyan'),
            (255, 255, 255, 'bright_white')
        ]
        
        # Find closest color
        min_distance = float('inf')
        closest_color = 'white'
        
        for cr, cg, cb, name in colors:
            distance = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
            if distance < min_distance:
                min_distance = distance
                closest_color = name
        
        return self.ANSI_COLORS[closest_color]
    
    def convert_to_ascii(self, image: Image.Image) -> List[str]:
        """
        Convert image to high-quality ASCII art.
        
        Args:
            image: PIL Image object
            
        Returns:
            List of ASCII art lines
        """
        # Store original for color extraction if needed
        original_rgb = image.convert('RGB') if self.use_color else None
        
        # Advanced preprocessing
        processed_img = self.advanced_preprocess_pil(image)
        
        # Resize with high-quality resampling
        new_width, new_height = self.calculate_output_size(processed_img)
        resized_img = processed_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Also resize color image if using color
        if self.use_color and original_rgb:
            resized_color = original_rgb.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to numpy for dithering
        img_array = np.array(resized_img)
        
        # Apply Floyd-Steinberg dithering
        dithered_array = self.floyd_steinberg_dither(img_array)
        
        # Convert to ASCII
        ascii_lines = []
        height, width = dithered_array.shape
        
        for row in range(height):
            line = ""
            for col in range(width):
                pixel_value = int(dithered_array[row, col])
                char = self.pixel_to_char_advanced(pixel_value)
                
                if self.use_color and original_rgb and self.output_format == 'ansi':
                    # Get color from original image
                    r, g, b = resized_color.getpixel((col, row))
                    color_code = self.rgb_to_ansi_color(r, g, b)
                    char = f"{color_code}{char}{self.ANSI_COLORS['reset']}"
                elif self.use_color and self.output_format == 'html':
                    # HTML color formatting will be handled in convert_to_html
                    pass
                
                line += char
            ascii_lines.append(line)
        
        return ascii_lines
    
    def convert_to_html(self, ascii_lines: List[str], image: Image.Image = None) -> str:
        """
        Convert ASCII lines to HTML format with optional color.
        
        Args:
            ascii_lines: List of ASCII art lines
            image: Original image for color extraction
            
        Returns:
            HTML string
        """
        html_lines = ['<!DOCTYPE html>']
        html_lines.append('<html><head><title>ASCII Art</title>')
        html_lines.append('<style>')
        html_lines.append('body { background: black; color: white; font-family: "Courier New", monospace; }')
        html_lines.append('pre { font-size: 2px; line-height: 2px; letter-spacing: 0px; }')
        html_lines.append('</style></head><body><pre>')
        
        if self.use_color and image:
            # Color version - need to process with color
            resized_color = image.convert('RGB').resize(
                (len(ascii_lines[0]), len(ascii_lines)), 
                Image.Resampling.LANCZOS
            )
            
            for row, line in enumerate(ascii_lines):
                html_line = ""
                for col, char in enumerate(line):
                    if col < resized_color.width and row < resized_color.height:
                        r, g, b = resized_color.getpixel((col, row))
                        color = f"rgb({r},{g},{b})"
                        html_line += f'<span style="color:{color}">{html.escape(char)}</span>'
                    else:
                        html_line += html.escape(char)
                html_lines.append(html_line)
        else:
            # Monochrome version
            for line in ascii_lines:
                html_lines.append(html.escape(line))
        
        html_lines.append('</pre></body></html>')
        return '\n'.join(html_lines)
    
    def convert_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert image file to high-quality ASCII art.
        
        Args:
            input_path: Path to input image
            output_path: Optional output file path
            
        Returns:
            ASCII art string (or HTML string)
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Image file not found: {input_path}")
        
        # Load image
        try:
            image = Image.open(input_path)
        except Exception as e:
            raise ValueError(f"Could not load image {input_path}: {e}")
        
        # Convert to ASCII
        ascii_lines = self.convert_to_ascii(image)
        
        # Format output based on type
        if self.output_format == 'html':
            output_content = self.convert_to_html(ascii_lines, image if self.use_color else None)
            extension = '.html'
        else:
            output_content = '\n'.join(ascii_lines)
            extension = '.txt'
        
        # Save to file if output path provided
        if output_path:
            if not output_path.endswith(('.txt', '.html')):
                output_path += extension
            
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(output_content)
                print(f"High-quality ASCII art saved to: {output_path}")
            except Exception as e:
                raise IOError(f"Could not save ASCII art to {output_path}: {e}")
        
        return output_content
    
    def get_info(self, image_path: str) -> dict:
        """Get conversion information."""
        image = Image.open(image_path)
        output_width, output_height = self.calculate_output_size(image)
        
        return {
            'input_size': image.size,
            'output_size': (output_width, output_height),
            'total_characters': output_width * output_height,
            'character_set': self.chars[:10] + '...' if len(self.chars) > 10 else self.chars,
            'output_format': self.output_format,
            'use_color': self.use_color,
            'quality_features': [
                'Ultra high resolution (300+ chars)',
                'Floyd-Steinberg dithering',
                'Histogram equalization',
                'Advanced edge detection',
                'Gamma correction',
                'Unsharp masking',
                'Perceptual brightness mapping'
            ]
        }


def create_4k_ascii(image_path: str, output_path: str = None, 
                   style: str = 'ultra', format_type: str = 'text', 
                   use_color: bool = False) -> str:
    """
    Create 4K-quality ASCII art with simplified approach.
    
    Args:
        image_path: Input image path
        output_path: Output file path
        style: 'ultra', 'fine', 'blocks', or 'sub_pixel'
        format_type: 'text', 'html', or 'ansi'
        use_color: Whether to use color
        
    Returns:
        ASCII art string
    """
    char_sets = {
        'ultra': 'ascii_ultra',
        'fine': 'ultra_fine', 
        'blocks': 'sub_pixel',
        'sub_pixel': 'sub_pixel'
    }
    
    widths = {
        'ultra': 350,
        'fine': 300,
        'blocks': 280,
        'sub_pixel': 200  # Blocks are denser, need fewer
    }
    
    converter = SimplifiedHighQualityASCII(
        char_set=char_sets.get(style, 'ascii_ultra'),
        width=widths.get(style, 320),
        output_format=format_type,
        use_color=use_color
    )
    
    return converter.convert_file(image_path, output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simplified High-Quality ASCII Art Converter")
    parser.add_argument('input', help='Input image path')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-s', '--style', choices=['ultra', 'fine', 'blocks', 'sub_pixel'], 
                       default='ultra', help='ASCII style')
    parser.add_argument('-f', '--format', choices=['text', 'html', 'ansi'], 
                       default='text', help='Output format')
    parser.add_argument('-c', '--color', action='store_true', help='Use color (HTML/ANSI only)')
    parser.add_argument('--info', action='store_true', help='Show conversion info')
    
    args = parser.parse_args()
    
    if args.info:
        converter = SimplifiedHighQualityASCII()
        info = converter.get_info(args.input)
        print(f"Conversion info for: {args.input}")
        for key, value in info.items():
            print(f"{key}: {value}")
    else:
        result = create_4k_ascii(args.input, args.output, args.style, args.format, args.color)
        if not args.output:
            print(result[:2000] + "..." if len(result) > 2000 else result)