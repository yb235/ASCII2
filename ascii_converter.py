#!/usr/bin/env python3
"""
ASCII Art Converter - High Quality Image to ASCII Art Conversion

This module provides advanced ASCII art conversion capabilities with multiple
quality enhancement techniques including edge detection, contrast adjustment,
and intelligent character mapping.
"""

import os
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Tuple, Optional, List
import argparse


class ASCIIConverter:
    """
    Advanced ASCII art converter with multiple quality enhancement techniques.
    """
    
    # Character sets ordered by density (light to dark)
    CHAR_SETS = {
        'simple': " .:-=+*#%@",
        'extended': " .`',:;\"^~-_+<>i!lI?/\\|()1{}[]rcvunxzjftLCJUYXZO0Qoahkbdpqwm*WMB8&%$#@",
        'blocks': " ░▒▓█",
        'detailed': " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    }
    
    def __init__(self, char_set: str = 'extended', width: int = 120, 
                 enhance_contrast: bool = True, edge_detection: bool = True):
        """
        Initialize ASCII converter with quality settings.
        
        Args:
            char_set: Character set to use ('simple', 'extended', 'blocks', 'detailed')
            width: Output width in characters
            enhance_contrast: Whether to enhance image contrast
            edge_detection: Whether to apply edge detection for better detail
        """
        self.chars = self.CHAR_SETS.get(char_set, self.CHAR_SETS['extended'])
        self.width = width
        self.enhance_contrast = enhance_contrast
        self.edge_detection = edge_detection
        self.char_len = len(self.chars)
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Apply preprocessing to enhance ASCII conversion quality.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to grayscale
        gray_img = image.convert('L')
        
        # Enhance contrast if enabled
        if self.enhance_contrast:
            enhancer = ImageEnhance.Contrast(gray_img)
            gray_img = enhancer.enhance(1.5)  # Increase contrast by 50%
        
        # Apply edge detection for better detail preservation
        if self.edge_detection:
            # Create edge-enhanced version
            edges = gray_img.filter(ImageFilter.FIND_EDGES)
            # Blend original with edges for better detail
            gray_img = Image.blend(gray_img, edges, 0.3)
        
        # Apply subtle sharpening
        gray_img = gray_img.filter(ImageFilter.SHARPEN)
        
        return gray_img
    
    def calculate_output_size(self, image: Image.Image) -> Tuple[int, int]:
        """
        Calculate optimal output dimensions maintaining aspect ratio.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (width, height) for ASCII output
        """
        orig_width, orig_height = image.size
        aspect_ratio = orig_height / orig_width
        
        # Account for character aspect ratio (characters are taller than wide)
        char_aspect_ratio = 0.5  # Typical character height/width ratio
        adjusted_height = int(self.width * aspect_ratio * char_aspect_ratio)
        
        return self.width, adjusted_height
    
    def resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize image to optimal dimensions for ASCII conversion.
        
        Args:
            image: PIL Image object
            
        Returns:
            Resized PIL Image
        """
        new_width, new_height = self.calculate_output_size(image)
        
        # Use high-quality resampling
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized
    
    def pixel_to_char(self, pixel_value: int) -> str:
        """
        Convert pixel brightness to ASCII character.
        
        Args:
            pixel_value: Pixel intensity (0-255)
            
        Returns:
            ASCII character representing the pixel
        """
        # Normalize pixel value to character index
        char_index = int((pixel_value / 255) * (self.char_len - 1))
        return self.chars[char_index]
    
    def convert_to_ascii(self, image: Image.Image) -> List[str]:
        """
        Convert image to ASCII art lines.
        
        Args:
            image: PIL Image object
            
        Returns:
            List of strings, each representing a line of ASCII art
        """
        # Preprocess image
        processed_img = self.preprocess_image(image)
        
        # Resize to target dimensions
        resized_img = self.resize_image(processed_img)
        
        # Convert to numpy array for efficient processing
        img_array = np.array(resized_img)
        
        # Convert each pixel to ASCII character
        ascii_lines = []
        height, width = img_array.shape
        
        for row in range(height):
            line = ""
            for col in range(width):
                pixel_value = img_array[row, col]
                line += self.pixel_to_char(pixel_value)
            ascii_lines.append(line)
        
        return ascii_lines
    
    def convert_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert image file to ASCII art and optionally save to file.
        
        Args:
            input_path: Path to input image file
            output_path: Optional path to save ASCII art. If None, returns ASCII as string.
            
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
                print(f"ASCII art saved to: {output_path}")
            except Exception as e:
                raise IOError(f"Could not save ASCII art to {output_path}: {e}")
        
        return ascii_art
    
    def get_conversion_info(self, image_path: str) -> dict:
        """
        Get information about the conversion settings and image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with conversion information
        """
        image = Image.open(image_path)
        output_width, output_height = self.calculate_output_size(image)
        
        return {
            'input_size': image.size,
            'output_size': (output_width, output_height),
            'character_set': self.chars,
            'settings': {
                'width': self.width,
                'enhance_contrast': self.enhance_contrast,
                'edge_detection': self.edge_detection
            }
        }


def create_enhanced_ascii_converter(image_path: str, output_path: str, quality: str = 'high') -> dict:
    """
    Create high-quality ASCII art with optimized settings.
    
    Args:
        image_path: Path to input image
        output_path: Path to save ASCII output
        quality: Quality level ('low', 'medium', 'high', 'ultra')
        
    Returns:
        Dictionary with conversion results and metadata
    """
    # Quality presets
    quality_settings = {
        'low': {
            'char_set': 'simple',
            'width': 60,
            'enhance_contrast': False,
            'edge_detection': False
        },
        'medium': {
            'char_set': 'extended',
            'width': 80,
            'enhance_contrast': True,
            'edge_detection': False
        },
        'high': {
            'char_set': 'detailed',
            'width': 120,
            'enhance_contrast': True,
            'edge_detection': True
        },
        'ultra': {
            'char_set': 'detailed',
            'width': 160,
            'enhance_contrast': True,
            'edge_detection': True
        }
    }
    
    settings = quality_settings.get(quality, quality_settings['high'])
    
    # Create converter with quality settings
    converter = ASCIIConverter(**settings)
    
    # Get conversion info
    info = converter.get_conversion_info(image_path)
    
    # Convert image
    ascii_art = converter.convert_file(image_path, output_path)
    
    # Return results with metadata
    return {
        'ascii_art': ascii_art,
        'conversion_info': info,
        'quality_level': quality,
        'output_file': output_path,
        'character_count': len(ascii_art),
        'line_count': len(ascii_art.split('\n'))
    }


def main():
    """Command line interface for ASCII converter."""
    parser = argparse.ArgumentParser(
        description="Convert images to high-quality ASCII art",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ascii_converter.py image.jpg                     # Basic conversion
    python ascii_converter.py image.jpg -o ascii_art.txt    # Save to file
    python ascii_converter.py image.jpg -q ultra -w 200     # Ultra quality
    python ascii_converter.py image.jpg -c blocks           # Use block characters
        """
    )
    
    parser.add_argument('input', help='Input image file path')
    parser.add_argument('-o', '--output', help='Output ASCII art file path')
    parser.add_argument('-w', '--width', type=int, default=120, 
                       help='Output width in characters (default: 120)')
    parser.add_argument('-q', '--quality', choices=['low', 'medium', 'high', 'ultra'],
                       default='high', help='Quality preset (default: high)')
    parser.add_argument('-c', '--charset', choices=['simple', 'extended', 'blocks', 'detailed'],
                       default='detailed', help='Character set to use (default: detailed)')
    parser.add_argument('--no-contrast', action='store_true', 
                       help='Disable contrast enhancement')
    parser.add_argument('--no-edges', action='store_true',
                       help='Disable edge detection')
    parser.add_argument('--info', action='store_true',
                       help='Show conversion information')
    
    args = parser.parse_args()
    
    try:
        # Create converter with custom settings if provided
        if args.quality in ['low', 'medium', 'high', 'ultra']:
            result = create_enhanced_ascii_converter(
                args.input, 
                args.output or 'ascii_output.txt', 
                args.quality
            )
            
            if args.info:
                print(f"Conversion completed successfully!")
                print(f"Quality level: {result['quality_level']}")
                print(f"Output size: {result['conversion_info']['output_size']}")
                print(f"Character count: {result['character_count']}")
                print(f"Line count: {result['line_count']}")
                
            if not args.output:
                print("\nASCII Art:")
                print("-" * 50)
                print(result['ascii_art'])
        else:
            # Custom settings
            converter = ASCIIConverter(
                char_set=args.charset,
                width=args.width,
                enhance_contrast=not args.no_contrast,
                edge_detection=not args.no_edges
            )
            
            ascii_art = converter.convert_file(args.input, args.output)
            
            if args.info:
                info = converter.get_conversion_info(args.input)
                print(f"Input size: {info['input_size']}")
                print(f"Output size: {info['output_size']}")
                print(f"Character set: {args.charset}")
            
            if not args.output:
                print("\nASCII Art:")
                print("-" * 50)
                print(ascii_art)
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())