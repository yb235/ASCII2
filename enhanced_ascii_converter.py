#!/usr/bin/env python3
"""
Enhanced ASCII Art Converter - Ultra High Quality 4K-Like ASCII Art

This module provides advanced ASCII art conversion with 4K-like resolution and quality,
featuring sophisticated preprocessing, advanced character mapping, and error diffusion.
"""

import os
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from typing import Tuple, Optional, List
import cv2


class EnhancedASCIIConverter:
    """
    Ultra high-quality ASCII art converter with 4K-like resolution capabilities.
    """
    
    # Ultra-detailed character sets with precise density gradations
    CHAR_SETS = {
        'ultra_detailed': " `.'\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
        'blocks_hd': " ░▒▓█",
        'sub_pixel': " ․‥⋯⋰⋱⋮⋯▁▂▃▄▅▆▇█",
        'gradients': " ░░▒▒▓▓██",
        'unicode_blocks': " ▘▝▀▖▌▞▛▗▚▐▜▄▙▟█",
        'braille': "⠀⠁⠂⠃⠄⠅⠆⠇⠈⠉⠊⠋⠌⠍⠎⠏⠐⠑⠒⠓⠔⠕⠖⠗⠘⠙⠚⠛⠜⠝⠞⠟⠠⠡⠢⠣⠤⠥⠦⠧⠨⠩⠪⠫⠬⠭⠮⠯⠰⠱⠲⠳⠴⠵⠶⠷⠸⠹⠺⠻⠼⠽⠾⠿",
        'ascii_extended': " .`',:;\"^~-_+<>i!lI?/\\|()1{}[]rcvunxzjftLCJUYXZO0Qoahkbdpqwm*WMB8&%$#@"
    }
    
    def __init__(self, char_set: str = 'ultra_detailed', width: int = 280, 
                 enhance_contrast: bool = True, edge_detection: bool = True,
                 gamma_correction: float = 1.2, use_dithering: bool = True,
                 histogram_equalization: bool = True):
        """
        Initialize enhanced ASCII converter with 4K-quality settings.
        
        Args:
            char_set: Character set to use
            width: Output width in characters (280+ for 4K-like quality)
            enhance_contrast: Whether to enhance image contrast
            edge_detection: Whether to apply edge detection
            gamma_correction: Gamma correction factor (1.0-2.0)
            use_dithering: Whether to use Floyd-Steinberg dithering
            histogram_equalization: Whether to apply histogram equalization
        """
        self.chars = self.CHAR_SETS.get(char_set, self.CHAR_SETS['ultra_detailed'])
        self.width = width
        self.enhance_contrast = enhance_contrast
        self.edge_detection = edge_detection
        self.gamma_correction = gamma_correction
        self.use_dithering = use_dithering
        self.histogram_equalization = histogram_equalization
        self.char_len = len(self.chars)
        
        # Precise character aspect ratio for modern terminals
        self.char_aspect_ratio = 0.45  # More accurate for most fonts
    
    def advanced_preprocess(self, image: Image.Image) -> Image.Image:
        """
        Apply advanced preprocessing for ultra-high quality conversion.
        
        Args:
            image: PIL Image object
            
        Returns:
            Heavily processed PIL Image optimized for ASCII conversion
        """
        # Convert to RGB first if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to OpenCV format for advanced processing
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        cv_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Convert back to PIL
        image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        # Convert to grayscale with weighted conversion for better contrast
        gray_img = image.convert('L')
        
        # Apply histogram equalization if enabled
        if self.histogram_equalization:
            gray_array = np.array(gray_img)
            gray_array = cv2.equalizeHist(gray_array)
            gray_img = Image.fromarray(gray_array)
        
        # Gamma correction for better mid-tone detail
        if self.gamma_correction != 1.0:
            gamma_table = [((i / 255.0) ** (1.0 / self.gamma_correction)) * 255 
                          for i in range(256)]
            gray_img = gray_img.point(gamma_table)
        
        # Enhanced contrast adjustment
        if self.enhance_contrast:
            enhancer = ImageEnhance.Contrast(gray_img)
            gray_img = enhancer.enhance(1.8)  # Stronger contrast boost
        
        # Advanced edge detection and preservation
        if self.edge_detection:
            # Use multiple edge detection methods
            edges1 = gray_img.filter(ImageFilter.FIND_EDGES)
            
            # Convert to OpenCV for Canny edge detection
            gray_cv = cv2.cvtColor(np.array(gray_img), cv2.COLOR_GRAY2BGR)
            gray_cv = cv2.cvtColor(gray_cv, cv2.COLOR_BGR2GRAY)
            edges2 = cv2.Canny(gray_cv, 50, 150)
            edges2 = Image.fromarray(edges2)
            
            # Combine edge detection methods
            combined_edges = Image.blend(edges1, edges2, 0.5)
            
            # Blend with original for detail preservation
            gray_img = Image.blend(gray_img, combined_edges, 0.25)
        
        # Apply unsharp masking for crisp details
        blurred = gray_img.filter(ImageFilter.GaussianBlur(1.0))
        unsharp_mask = ImageEnhance.Sharpness(gray_img).enhance(2.0)
        gray_img = Image.blend(gray_img, unsharp_mask, 0.3)
        
        # Final sharpening
        gray_img = gray_img.filter(ImageFilter.SHARPEN)
        
        return gray_img
    
    def calculate_output_size(self, image: Image.Image) -> Tuple[int, int]:
        """
        Calculate optimal output dimensions for 4K-like quality.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (width, height) for ultra high-res ASCII output
        """
        orig_width, orig_height = image.size
        aspect_ratio = orig_height / orig_width
        
        # Calculate height with precise character aspect ratio
        adjusted_height = int(self.width * aspect_ratio * self.char_aspect_ratio)
        
        # Ensure minimum height for detail
        if adjusted_height < 60:
            adjusted_height = 60
            
        return self.width, adjusted_height
    
    def floyd_steinberg_dither(self, image_array: np.ndarray) -> np.ndarray:
        """
        Apply Floyd-Steinberg dithering for smoother gradients.
        
        Args:
            image_array: 2D numpy array of grayscale values
            
        Returns:
            Dithered array with error diffusion
        """
        height, width = image_array.shape
        dithered = image_array.astype(float)
        
        for y in range(height):
            for x in range(width):
                old_pixel = dithered[y, x]
                new_pixel = round(old_pixel * (self.char_len - 1) / 255) * 255 / (self.char_len - 1)
                dithered[y, x] = new_pixel
                
                error = old_pixel - new_pixel
                
                # Distribute error to neighboring pixels
                if x + 1 < width:
                    dithered[y, x + 1] += error * 7/16
                if y + 1 < height:
                    if x > 0:
                        dithered[y + 1, x - 1] += error * 3/16
                    dithered[y + 1, x] += error * 5/16
                    if x + 1 < width:
                        dithered[y + 1, x + 1] += error * 1/16
        
        return np.clip(dithered, 0, 255)
    
    def adaptive_pixel_to_char(self, pixel_value: int, local_contrast: float = 1.0) -> str:
        """
        Convert pixel brightness to ASCII character with local adaptation.
        
        Args:
            pixel_value: Pixel intensity (0-255)
            local_contrast: Local contrast factor for adaptive mapping
            
        Returns:
            ASCII character representing the pixel
        """
        # Apply local contrast adjustment
        adjusted_value = pixel_value * local_contrast
        adjusted_value = np.clip(adjusted_value, 0, 255)
        
        # Non-linear mapping for better visual perception
        normalized = adjusted_value / 255.0
        # Apply power curve for better perceptual mapping
        perceptual_value = normalized ** 0.8
        
        char_index = int(perceptual_value * (self.char_len - 1))
        char_index = np.clip(char_index, 0, self.char_len - 1)
        
        return self.chars[char_index]
    
    def calculate_local_contrast(self, image_array: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Calculate local contrast for adaptive character mapping.
        
        Args:
            image_array: 2D numpy array of grayscale values
            kernel_size: Size of local neighborhood
            
        Returns:
            Local contrast map
        """
        # Use OpenCV for efficient local statistics
        cv_image = image_array.astype(np.uint8)
        
        # Calculate local mean and standard deviation
        mean = cv2.blur(cv_image.astype(np.float32), (kernel_size, kernel_size))
        sqr_mean = cv2.blur((cv_image.astype(np.float32)) ** 2, (kernel_size, kernel_size))
        variance = sqr_mean - mean ** 2
        std_dev = np.sqrt(np.maximum(variance, 0))
        
        # Normalize standard deviation to get contrast factor
        contrast = std_dev / (mean + 1e-8)  # Avoid division by zero
        contrast = np.clip(contrast, 0.5, 2.0)  # Reasonable range
        
        return contrast
    
    def convert_to_ascii(self, image: Image.Image) -> List[str]:
        """
        Convert image to ultra high-quality ASCII art.
        
        Args:
            image: PIL Image object
            
        Returns:
            List of strings, each representing a line of ASCII art
        """
        # Advanced preprocessing
        processed_img = self.advanced_preprocess(image)
        
        # Resize to target dimensions with high-quality resampling
        new_width, new_height = self.calculate_output_size(processed_img)
        resized_img = processed_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(resized_img)
        
        # Apply dithering if enabled
        if self.use_dithering:
            img_array = self.floyd_steinberg_dither(img_array)
        
        # Calculate local contrast for adaptive mapping
        local_contrast = self.calculate_local_contrast(img_array)
        
        # Convert each pixel to ASCII character
        ascii_lines = []
        height, width = img_array.shape
        
        for row in range(height):
            line = ""
            for col in range(width):
                pixel_value = int(img_array[row, col])
                contrast_factor = local_contrast[row, col]
                char = self.adaptive_pixel_to_char(pixel_value, contrast_factor)
                line += char
            ascii_lines.append(line)
        
        return ascii_lines
    
    def convert_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert image file to ultra high-quality ASCII art.
        
        Args:
            input_path: Path to input image file
            output_path: Optional path to save ASCII art
            
        Returns:
            ASCII art as string
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
        ascii_art = "\n".join(ascii_lines)
        
        # Save to file if output path provided
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(ascii_art)
                print(f"Ultra high-quality ASCII art saved to: {output_path}")
            except Exception as e:
                raise IOError(f"Could not save ASCII art to {output_path}: {e}")
        
        return ascii_art
    
    def get_conversion_info(self, image_path: str) -> dict:
        """
        Get detailed information about the conversion settings.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with comprehensive conversion information
        """
        image = Image.open(image_path)
        output_width, output_height = self.calculate_output_size(image)
        total_chars = output_width * output_height
        
        return {
            'input_size': image.size,
            'output_size': (output_width, output_height),
            'total_characters': total_chars,
            'character_set': self.chars,
            'character_count': self.char_len,
            'settings': {
                'width': self.width,
                'enhance_contrast': self.enhance_contrast,
                'edge_detection': self.edge_detection,
                'gamma_correction': self.gamma_correction,
                'use_dithering': self.use_dithering,
                'histogram_equalization': self.histogram_equalization,
                'char_aspect_ratio': self.char_aspect_ratio
            },
            'quality_features': [
                'CLAHE histogram equalization',
                'Floyd-Steinberg dithering',
                'Multi-method edge detection',
                'Adaptive local contrast mapping',
                'Gamma correction',
                'Unsharp masking',
                'High-resolution output (4K-like)'
            ]
        }


def create_4k_ascii_art(image_path: str, output_path: str = None, 
                       style: str = 'ultra') -> str:
    """
    Create 4K-quality ASCII art with optimized settings.
    
    Args:
        image_path: Path to input image
        output_path: Optional output file path
        style: Quality style ('ultra', 'high', 'detailed')
        
    Returns:
        ASCII art string
    """
    if style == 'ultra':
        converter = EnhancedASCIIConverter(
            char_set='ultra_detailed',
            width=320,
            enhance_contrast=True,
            edge_detection=True,
            gamma_correction=1.3,
            use_dithering=True,
            histogram_equalization=True
        )
    elif style == 'detailed':
        converter = EnhancedASCIIConverter(
            char_set='braille',
            width=280,
            enhance_contrast=True,
            edge_detection=True,
            gamma_correction=1.2,
            use_dithering=True,
            histogram_equalization=True
        )
    else:  # high
        converter = EnhancedASCIIConverter(
            char_set='unicode_blocks',
            width=240,
            enhance_contrast=True,
            edge_detection=True,
            gamma_correction=1.1,
            use_dithering=True,
            histogram_equalization=False
        )
    
    return converter.convert_file(image_path, output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultra High-Quality ASCII Art Converter")
    parser.add_argument('input', help='Input image path')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-w', '--width', type=int, default=280, help='Output width (default: 280)')
    parser.add_argument('-s', '--style', choices=['ultra', 'high', 'detailed'], 
                       default='ultra', help='Quality style')
    parser.add_argument('--info', action='store_true', help='Show conversion info')
    
    args = parser.parse_args()
    
    if args.info:
        converter = EnhancedASCIIConverter(width=args.width)
        info = converter.get_conversion_info(args.input)
        print(f"Conversion Info for: {args.input}")
        print(f"Input size: {info['input_size']}")
        print(f"Output size: {info['output_size']}")
        print(f"Total characters: {info['total_characters']:,}")
        print(f"Quality features: {', '.join(info['quality_features'])}")
    else:
        ascii_art = create_4k_ascii_art(args.input, args.output, args.style)
        if not args.output:
            print(ascii_art)