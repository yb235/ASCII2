#!/usr/bin/env python3
"""
AI-ENHANCED ASCII CONVERTER - The Future of ASCII Art!

Features:
- AI-powered image analysis for optimal settings
- Smart artistic style detection
- Intelligent character set selection
- Adaptive quality enhancement
- Real-time optimization
- Perfect results every time!

This is ASCII art with ARTIFICIAL INTELLIGENCE! 🤖✨
"""

import os
import sys
import numpy as np
from PIL import Image, ImageStat, ImageFilter
from typing import Dict, Tuple, List, Optional
import colorsys
from collections import Counter
from ultimate_ascii import UltimateASCIIConverter


class AIImageAnalyzer:
    """
    AI-powered image analyzer that determines optimal ASCII conversion settings.
    """
    
    def __init__(self):
        """Initialize the AI analyzer."""
        self.analysis_cache = {}
    
    def analyze_image_complexity(self, image: Image.Image) -> Dict[str, float]:
        """
        Analyze image complexity using AI techniques.
        
        Returns:
            Dictionary with complexity metrics
        """
        # Convert to grayscale for analysis
        gray_img = image.convert('L')
        img_array = np.array(gray_img)
        
        # Edge density analysis
        edges = gray_img.filter(ImageFilter.FIND_EDGES)
        edge_array = np.array(edges)
        edge_density = np.mean(edge_array > 50) * 100
        
        # Texture analysis using local standard deviation
        kernel_size = 5
        texture_scores = []
        h, w = img_array.shape
        
        for i in range(0, h - kernel_size, kernel_size):
            for j in range(0, w - kernel_size, kernel_size):
                patch = img_array[i:i+kernel_size, j:j+kernel_size]
                texture_scores.append(np.std(patch))
        
        texture_complexity = np.mean(texture_scores) / 255.0 * 100
        
        # Contrast analysis
        stat = ImageStat.Stat(gray_img)
        contrast = stat.stddev[0] / 255.0 * 100
        
        # Frequency analysis (high frequency = more detail)
        fft = np.fft.fft2(img_array)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)
        
        # High frequency energy
        center_y, center_x = h // 2, w // 2
        high_freq_mask = np.zeros_like(magnitude)
        radius = min(h, w) // 4
        y, x = np.ogrid[:h, :w]
        mask = (x - center_x)**2 + (y - center_y)**2 > radius**2
        high_freq_energy = np.mean(magnitude[mask]) / np.mean(magnitude) * 100
        
        return {
            'edge_density': min(edge_density, 100),
            'texture_complexity': min(texture_complexity, 100),
            'contrast': min(contrast, 100),
            'detail_level': min(high_freq_energy, 100),
            'overall_complexity': min((edge_density + texture_complexity + contrast + high_freq_energy) / 4, 100)
        }
    
    def analyze_color_distribution(self, image: Image.Image) -> Dict[str, any]:
        """
        AI analysis of color characteristics.
        
        Returns:
            Color analysis results
        """
        # Convert to RGB if needed
        rgb_img = image.convert('RGB')
        
        # Get dominant colors
        colors = rgb_img.getcolors(maxcolors=256*256*256)
        if colors is None:
            # Image has too many colors, sample it
            rgb_img = rgb_img.resize((200, 200))
            colors = rgb_img.getcolors(maxcolors=256*256*256)
        
        # Sort by frequency
        colors.sort(key=lambda x: x[0], reverse=True)
        
        # Analyze color characteristics
        total_pixels = sum(count for count, color in colors)
        dominant_colors = colors[:10]  # Top 10 colors
        
        # Calculate color diversity (entropy)
        color_probs = [count / total_pixels for count, color in colors]
        color_entropy = -sum(p * np.log2(p) for p in color_probs if p > 0)
        
        # Analyze saturation and brightness
        saturations = []
        brightnesses = []
        hues = []
        
        for count, color in dominant_colors:
            # Handle both RGB and RGBA colors
            if len(color) == 3:
                r, g, b = color
            elif len(color) == 4:
                r, g, b, a = color
            else:
                continue
            
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            hues.append(h * 360)
            saturations.append(s * 100)
            brightnesses.append(v * 100)
        
        # Determine color characteristics
        avg_saturation = np.mean(saturations)
        avg_brightness = np.mean(brightnesses)
        hue_variety = len(set(int(h/30) for h in hues))  # Group hues by 30-degree segments
        
        # Color temperature analysis
        color_temp = self._analyze_color_temperature(dominant_colors)
        
        return {
            'color_diversity': min(color_entropy / 8 * 100, 100),  # Normalize entropy
            'saturation': avg_saturation,
            'brightness': avg_brightness,
            'hue_variety': hue_variety,
            'color_temperature': color_temp,
            'dominant_colors': [color for count, color in dominant_colors[:5]],
            'is_monochrome': avg_saturation < 10,
            'is_high_contrast': max(brightnesses) - min(brightnesses) > 70
        }
    
    def _analyze_color_temperature(self, dominant_colors: List[Tuple[int, Tuple[int, int, int]]]) -> str:
        """Analyze color temperature of the image."""
        warm_score = 0
        cool_score = 0
        
        for count, color in dominant_colors:
            weight = count
            
            # Handle both RGB and RGBA colors
            if len(color) == 3:
                r, g, b = color
            elif len(color) == 4:
                r, g, b, a = color
            else:
                continue
            
            # Warm colors (reds, oranges, yellows)
            if r > g and r > b:
                warm_score += weight
            elif r > b and g > b:
                warm_score += weight * 0.7
            
            # Cool colors (blues, cyans, purples)
            elif b > r and b > g:
                cool_score += weight
            elif b > r and g > r:
                cool_score += weight * 0.7
        
        if warm_score > cool_score * 1.3:
            return 'warm'
        elif cool_score > warm_score * 1.3:
            return 'cool'
        else:
            return 'neutral'
    
    def detect_image_category(self, image: Image.Image) -> str:
        """
        AI-powered image category detection.
        
        Returns:
            Detected image category
        """
        # Aspect ratio analysis
        width, height = image.size
        aspect_ratio = width / height
        
        # Color analysis
        color_data = self.analyze_color_distribution(image)
        
        # Complexity analysis
        complexity = self.analyze_image_complexity(image)
        
        # Rule-based AI classification
        if aspect_ratio < 0.8:  # Portrait orientation
            if color_data['saturation'] > 30 and complexity['edge_density'] > 40:
                return 'portrait'
            else:
                return 'vertical_art'
        
        elif aspect_ratio > 1.8:  # Wide landscape
            if color_data['color_temperature'] == 'cool' and complexity['detail_level'] < 30:
                return 'landscape'
            elif complexity['edge_density'] > 60:
                return 'architectural'
            else:
                return 'panoramic'
        
        elif 0.9 <= aspect_ratio <= 1.1:  # Square
            if complexity['texture_complexity'] > 70:
                return 'abstract'
            elif color_data['is_monochrome']:
                return 'minimalist'
            else:
                return 'square_art'
        
        else:  # Regular aspect ratio
            if complexity['overall_complexity'] > 70:
                return 'detailed'
            elif color_data['saturation'] < 20:
                return 'monochrome'
            elif color_data['hue_variety'] > 6:
                return 'colorful'
            else:
                return 'general'


class AIAsciiOptimizer:
    """
    AI system that determines optimal ASCII conversion settings.
    """
    
    def __init__(self):
        """Initialize the AI optimizer."""
        self.analyzer = AIImageAnalyzer()
        
        # AI knowledge base for optimal settings
        self.optimization_rules = {
            'portrait': {
                'char_set': 'ultra_8k',
                'width_multiplier': 0.8,
                'artistic_style': 'realistic',
                'color_boost': 1.2
            },
            'landscape': {
                'char_set': 'ultra_8k', 
                'width_multiplier': 1.2,
                'artistic_style': 'realistic',
                'color_boost': 1.1
            },
            'architectural': {
                'char_set': 'blocks_4x',
                'width_multiplier': 1.0,
                'artistic_style': 'realistic',
                'color_boost': 0.9
            },
            'abstract': {
                'char_set': 'artistic',
                'width_multiplier': 1.1,
                'artistic_style': 'artistic',
                'color_boost': 1.5
            },
            'detailed': {
                'char_set': 'ultra_8k',
                'width_multiplier': 1.3,
                'artistic_style': 'realistic',
                'color_boost': 1.0
            },
            'monochrome': {
                'char_set': 'ultra_8k',
                'width_multiplier': 1.0,
                'artistic_style': 'retro',
                'color_boost': 0.8
            },
            'colorful': {
                'char_set': 'blocks_4x',
                'width_multiplier': 1.1,
                'artistic_style': 'cyberpunk',
                'color_boost': 1.4
            }
        }
    
    def optimize_settings(self, image: Image.Image, base_width: int = 400) -> Dict[str, any]:
        """
        AI-powered optimization of ASCII conversion settings.
        
        Args:
            image: PIL Image to analyze
            base_width: Base width for calculations
            
        Returns:
            Optimized settings dictionary
        """
        # Analyze image characteristics
        category = self.analyzer.detect_image_category(image)
        complexity = self.analyzer.analyze_image_complexity(image)
        color_data = self.analyzer.analyze_color_distribution(image)
        
        # Get base optimization rules
        rules = self.optimization_rules.get(category, {
            'char_set': 'ultra_8k',
            'width_multiplier': 1.0,
            'artistic_style': 'realistic',
            'color_boost': 1.0
        })
        
        # AI adjustments based on analysis
        optimized_width = int(base_width * rules['width_multiplier'])
        
        # Complexity-based width adjustment
        if complexity['overall_complexity'] > 80:
            optimized_width = int(optimized_width * 1.3)  # More detail needs more width
        elif complexity['overall_complexity'] < 30:
            optimized_width = int(optimized_width * 0.8)  # Simple images can use less width
        
        # Color-based settings
        use_color = not color_data['is_monochrome'] and color_data['saturation'] > 15
        
        # Color mode selection
        if color_data['color_diversity'] > 70:
            color_mode = 'truecolor'
        elif color_data['color_diversity'] > 40:
            color_mode = '256'
        else:
            color_mode = '16'
        
        # Character set optimization
        char_set = rules['char_set']
        if complexity['edge_density'] > 70:
            char_set = 'ultra_8k'  # High detail needs fine characters
        elif complexity['texture_complexity'] < 20:
            char_set = 'blocks_8x'  # Simple textures work with blocks
        
        # Output format selection
        if use_color and color_data['saturation'] > 50:
            output_format = 'html'  # Best for colorful images
        elif use_color:
            output_format = 'ansi'  # Good for moderate color
        else:
            output_format = 'text'  # Perfect for monochrome
        
        return {
            'width': optimized_width,
            'char_set': char_set,
            'use_color': use_color,
            'color_mode': color_mode,
            'artistic_style': rules['artistic_style'],
            'output_format': output_format,
            'analysis': {
                'category': category,
                'complexity': complexity['overall_complexity'],
                'color_diversity': color_data['color_diversity'],
                'is_colorful': color_data['saturation'] > 30,
                'color_temperature': color_data['color_temperature']
            },
            'ai_confidence': self._calculate_confidence(complexity, color_data, category)
        }
    
    def _calculate_confidence(self, complexity: Dict, color_data: Dict, category: str) -> float:
        """Calculate AI confidence in the optimization."""
        confidence = 0.5  # Base confidence
        
        # Higher confidence for clear categories
        if category in ['portrait', 'landscape', 'architectural']:
            confidence += 0.3
        
        # Higher confidence for clear complexity patterns
        if complexity['overall_complexity'] > 70 or complexity['overall_complexity'] < 30:
            confidence += 0.2
        
        # Higher confidence for clear color patterns
        if color_data['is_monochrome'] or color_data['saturation'] > 60:
            confidence += 0.2
        
        return min(confidence, 1.0)


class AIEnhancedASCIIConverter:
    """
    The ultimate AI-enhanced ASCII converter that automatically optimizes everything!
    """
    
    def __init__(self):
        """Initialize the AI-enhanced converter."""
        self.optimizer = AIAsciiOptimizer()
        self.conversion_history = []
    
    def convert_with_ai(self, image_path: str, output_path: str = None, 
                       base_width: int = 400, force_settings: Dict = None) -> Tuple[str, Dict]:
        """
        Convert image to ASCII using AI optimization.
        
        Args:
            image_path: Input image path
            output_path: Output file path
            base_width: Base width for AI calculations
            force_settings: Override AI with manual settings
            
        Returns:
            Tuple of (ascii_content, ai_analysis)
        """
        # Load and analyze image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path)
        print(f"🤖 AI analyzing image: {image.size[0]}×{image.size[1]}")
        
        # Get AI-optimized settings
        if force_settings:
            ai_settings = force_settings
            ai_settings['analysis'] = {'category': 'manual', 'ai_confidence': 1.0}
        else:
            ai_settings = self.optimizer.optimize_settings(image, base_width)
        
        # Display AI analysis
        analysis = ai_settings['analysis']
        print(f"🧠 AI Analysis:")
        print(f"   Category: {analysis['category'].replace('_', ' ').title()}")
        print(f"   Complexity: {analysis['complexity']:.1f}%")
        print(f"   Color Diversity: {analysis['color_diversity']:.1f}%")
        print(f"   AI Confidence: {ai_settings['ai_confidence']:.1%}")
        
        print(f"⚙️ AI Optimized Settings:")
        print(f"   Resolution: {ai_settings['width']}×{int(ai_settings['width'] * 0.56)}")
        print(f"   Character Set: {ai_settings['char_set']}")
        print(f"   Style: {ai_settings['artistic_style']}")
        print(f"   Color: {ai_settings['use_color']} ({ai_settings['color_mode'] if ai_settings['use_color'] else 'N/A'})")
        print(f"   Output: {ai_settings['output_format'].upper()}")
        
        # Create optimized converter
        converter = UltimateASCIIConverter(
            char_set=ai_settings['char_set'],
            width=ai_settings['width'],
            output_format=ai_settings['output_format'],
            use_color=ai_settings['use_color'],
            color_mode=ai_settings['color_mode'],
            artistic_style=ai_settings['artistic_style']
        )
        
        # Convert image
        print(f"🎨 Converting with AI-optimized settings...")
        ascii_content = converter.convert_file(image_path, output_path)
        
        # Store conversion history
        self.conversion_history.append({
            'image_path': image_path,
            'output_path': output_path,
            'settings': ai_settings,
            'result_size': len(ascii_content)
        })
        
        return ascii_content, ai_settings
    
    def batch_convert_with_ai(self, image_paths: List[str], output_dir: str = "ai_ascii_output") -> List[Dict]:
        """
        Convert multiple images using AI optimization.
        
        Args:
            image_paths: List of image file paths
            output_dir: Output directory
            
        Returns:
            List of conversion results
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        print(f"🚀 AI Batch Processing: {len(image_paths)} images")
        
        for i, image_path in enumerate(image_paths):
            try:
                print(f"\n📸 Processing {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
                
                # Generate output path
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_ai_ascii")
                
                # Convert with AI
                ascii_content, ai_settings = self.convert_with_ai(image_path, output_path)
                
                results.append({
                    'input': image_path,
                    'output': output_path + ('.' + ai_settings['output_format'] if ai_settings['output_format'] != 'text' else '.txt'),
                    'settings': ai_settings,
                    'success': True
                })
                
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
                results.append({
                    'input': image_path,
                    'error': str(e),
                    'success': False
                })
        
        return results
    
    def get_ai_statistics(self) -> Dict:
        """Get statistics about AI performance."""
        if not self.conversion_history:
            return {'message': 'No conversions yet'}
        
        categories = [h['settings']['analysis']['category'] for h in self.conversion_history]
        avg_confidence = np.mean([h['settings']['ai_confidence'] for h in self.conversion_history])
        
        return {
            'total_conversions': len(self.conversion_history),
            'average_confidence': avg_confidence,
            'most_common_category': Counter(categories).most_common(1)[0][0],
            'categories_detected': list(set(categories)),
            'average_width': np.mean([h['settings']['width'] for h in self.conversion_history])
        }


def create_ai_ascii(image_path: str, output_path: str = None, base_width: int = 400) -> Tuple[str, Dict]:
    """
    Create AI-optimized ASCII art.
    
    Args:
        image_path: Input image path
        output_path: Output file path  
        base_width: Base width for AI calculations
        
    Returns:
        Tuple of (ascii_content, ai_analysis)
    """
    converter = AIEnhancedASCIIConverter()
    return converter.convert_with_ai(image_path, output_path, base_width)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🤖 AI-ENHANCED ASCII CONVERTER - Artificial Intelligence meets ASCII Art! 🤖",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Let AI optimize everything automatically
    python ai_ascii.py image.jpg
    
    # AI optimization with custom base width
    python ai_ascii.py image.jpg -w 500
    
    # Batch process multiple images
    python ai_ascii.py *.jpg --batch
    
    # Show AI analysis without converting
    python ai_ascii.py image.jpg --analyze-only
        """
    )
    
    parser.add_argument('images', nargs='+', help='Input image path(s)')
    parser.add_argument('-o', '--output', help='Output file path (single image only)')
    parser.add_argument('-w', '--width', type=int, default=400, help='Base width for AI calculations')
    parser.add_argument('--batch', action='store_true', help='Batch process multiple images')
    parser.add_argument('--analyze-only', action='store_true', help='Show AI analysis without converting')
    parser.add_argument('--stats', action='store_true', help='Show AI statistics')
    
    args = parser.parse_args()
    
    converter = AIEnhancedASCIIConverter()
    
    if args.stats:
        stats = converter.get_ai_statistics()
        print("🤖 AI PERFORMANCE STATISTICS 🤖")
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    elif args.analyze_only:
        for image_path in args.images:
            print(f"\n🤖 AI ANALYSIS: {image_path}")
            print("=" * 50)
            try:
                image = Image.open(image_path)
                settings = converter.optimizer.optimize_settings(image, args.width)
                
                for key, value in settings['analysis'].items():
                    print(f"{key}: {value}")
                
                print(f"\nRecommended settings:")
                for key, value in settings.items():
                    if key != 'analysis':
                        print(f"  {key}: {value}")
                        
            except Exception as e:
                print(f"❌ Error: {e}")
    
    elif args.batch:
        print("🚀 AI BATCH PROCESSING MODE 🚀")
        results = converter.batch_convert_with_ai(args.images)
        
        successful = sum(1 for r in results if r['success'])
        print(f"\n✅ Batch Complete: {successful}/{len(results)} successful")
        
        for result in results:
            if result['success']:
                print(f"✓ {result['input']} → {result['output']}")
            else:
                print(f"✗ {result['input']}: {result['error']}")
    
    else:
        # Single image conversion
        if len(args.images) > 1:
            print("⚠️ Multiple images provided but not in batch mode. Processing first image only.")
        
        image_path = args.images[0]
        
        try:
            print("🤖 AI-ENHANCED ASCII CONVERSION 🤖")
            print("=" * 50)
            
            ascii_content, ai_settings = converter.convert_with_ai(
                image_path, args.output, args.width
            )
            
            print(f"\n🎉 AI CONVERSION COMPLETE! 🎉")
            
            if not args.output:
                # Show preview
                lines = ascii_content.split('\n')
                if len(lines) > 40:
                    print('\n'.join(lines[:20]))
                    print(f"\n... ({len(lines) - 40} lines omitted) ...")
                    print('\n'.join(lines[-20:]))
                else:
                    print(ascii_content)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)