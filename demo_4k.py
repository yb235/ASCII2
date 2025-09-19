#!/usr/bin/env python3
"""
4K ASCII Art Demonstration Script

This script demonstrates the new 4K resolution ASCII art capability,
showing the quality improvements over standard resolution levels.
"""

import os
import time
from ascii_converter import create_enhanced_ascii_converter

def demo_4k_ascii():
    """Demonstrate 4K ASCII art generation and quality comparison."""
    
    print("🎨 4K ASCII Art Demonstration")
    print("=" * 50)
    
    # Check if sample image exists
    sample_image = "WIN_20250919_19_52_29_Pro.jpg"
    if not os.path.exists(sample_image):
        print(f"❌ Sample image not found: {sample_image}")
        return False
    
    print(f"📸 Processing: {sample_image}")
    print()
    
    # Test all quality levels
    qualities = ['medium', 'high', 'ultra', '4k']
    results = []
    
    for quality in qualities:
        print(f"🔄 Generating {quality.upper()} quality ASCII...")
        
        start_time = time.time()
        try:
            result = create_enhanced_ascii_converter(
                sample_image,
                f"demo_{quality}_ascii.txt",
                quality
            )
            processing_time = time.time() - start_time
            
            width, height = result['conversion_info']['output_size']
            char_count = result['character_count']
            
            results.append({
                'quality': quality,
                'width': width,
                'height': height,
                'characters': char_count,
                'time': processing_time,
                'file': f"demo_{quality}_ascii.txt"
            })
            
            print(f"  ✓ {width}x{height} characters ({char_count:,} total)")
            print(f"  ⏱️  Processing time: {processing_time:.2f}s")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
        
        print()
    
    # Display comparison
    print("📊 QUALITY COMPARISON")
    print("-" * 50)
    print(f"{'Quality':<8} {'Dimensions':<12} {'Characters':<12} {'Time':<8}")
    print("-" * 50)
    
    for result in results:
        dimensions = f"{result['width']}x{result['height']}"
        characters = f"{result['characters']:,}"
        time_str = f"{result['time']:.2f}s"
        print(f"{result['quality']:<8} {dimensions:<12} {characters:<12} {time_str:<8}")
    
    # Show 4K improvements
    if len(results) >= 4:
        ultra_chars = results[2]['characters']  # ultra is index 2
        four_k_chars = results[3]['characters']  # 4k is index 3
        improvement = four_k_chars / ultra_chars
        
        print()
        print("🚀 4K IMPROVEMENTS")
        print("-" * 30)
        print(f"4K provides {improvement:.1f}x more detail than Ultra quality")
        print(f"4K character density: {four_k_chars:,} characters")
        print(f"Perfect for high-resolution displays and fine detail work")
    
    print()
    print("📁 Generated Files:")
    for result in results:
        file_size = os.path.getsize(result['file']) if os.path.exists(result['file']) else 0
        print(f"  - {result['file']} ({file_size:,} bytes)")
    
    print()
    print("✨ 4K ASCII demonstration completed!")
    print("Use 'python ascii_converter.py image.jpg -q 4k' for 4K conversion")
    
    return True

if __name__ == "__main__":
    success = demo_4k_ascii()
    exit(0 if success else 1)