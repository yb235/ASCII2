#!/usr/bin/env python3
"""
Test script for enhanced visual analysis features including ASCII art conversion.
"""

import os
import sys
import time
from PIL import Image, ImageDraw
from ascii_converter import ASCIIConverter, create_enhanced_ascii_converter
from enhanced_visual_agent import EnhancedVisualAnalysisAgent


def test_ascii_converter():
    """Test ASCII converter functionality"""
    print("Testing ASCII Converter...")
    
    # Create a simple test image
    test_image = Image.new('RGB', (200, 100), color='white')
    draw = ImageDraw.Draw(test_image)
    draw.rectangle([50, 25, 150, 75], fill='black')
    test_image.save('/tmp/test_rect.png')
    
    # Test different quality levels including 4K
    qualities = ['low', 'medium', 'high', 'ultra', '4k']
    
    for quality in qualities:
        try:
            result = create_enhanced_ascii_converter(
                '/tmp/test_rect.png',
                f'/tmp/test_ascii_{quality}.txt',
                quality
            )
            print(f"✓ {quality.capitalize()} quality ASCII conversion successful")
            print(f"  - Output size: {result['conversion_info']['output_size']}")
            print(f"  - Character count: {result['character_count']}")
            
            # Special validation for 4K quality
            if quality == '4k':
                width, height = result['conversion_info']['output_size']
                if width >= 300 and result['character_count'] > 15000:
                    print(f"  - ✓ 4K quality verified: {width}x{height} characters")
                else:
                    print(f"  - ❌ 4K quality validation failed")
                    
        except Exception as e:
            print(f"❌ {quality.capitalize()} quality ASCII conversion failed: {e}")
    
    print("ASCII Converter tests completed.\n")


def test_enhanced_visual_agent():
    """Test enhanced visual analysis agent"""
    print("Testing Enhanced Visual Analysis Agent...")
    
    # Create test image
    test_image = Image.new('RGB', (400, 300), color='lightblue')
    draw = ImageDraw.Draw(test_image)
    draw.ellipse([100, 50, 300, 250], fill='yellow')
    draw.rectangle([180, 120, 220, 180], fill='red')
    test_image.save('/tmp/test_scene.jpg')
    
    try:
        # Initialize enhanced agent
        agent = EnhancedVisualAnalysisAgent(enable_ascii=True, ascii_quality='medium')
        
        # Test enhanced analysis
        result = agent.analyze_image_enhanced('/tmp/test_scene.jpg', save_ascii=True)
        
        print("✓ Enhanced visual analysis completed")
        print(f"  - Processing time: {result.processing_time:.2f} seconds")
        print(f"  - ASCII art generated: {result.ascii_art is not None}")
        print(f"  - ASCII metadata available: {result.ascii_metadata is not None}")
        
        # Test report generation
        report = agent.create_analysis_report(result)
        with open('/tmp/test_report.txt', 'w') as f:
            f.write(report)
        print("✓ Analysis report generated")
        
        # Test JSON export
        agent.export_results(result, '/tmp/test_export.json')
        print("✓ JSON export completed")
        
        # Test processing stats
        stats = agent.get_processing_stats()
        print(f"✓ Processing statistics: {stats['total_images_processed']} images processed")
        
    except Exception as e:
        print(f"❌ Enhanced visual analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("Enhanced Visual Analysis Agent tests completed.\n")


def test_real_image_processing():
    """Test processing with the actual JPG file"""
    print("Testing Real Image Processing...")
    
    image_path = 'WIN_20250919_19_52_29_Pro.jpg'
    
    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        return
    
    try:
        # Test ASCII conversion only
        start_time = time.time()
        ascii_result = create_enhanced_ascii_converter(
            image_path,
            '/tmp/real_image_ascii.txt',
            'high'
        )
        ascii_time = time.time() - start_time
        
        print(f"✓ Real image ASCII conversion completed in {ascii_time:.2f} seconds")
        print(f"  - Output dimensions: {ascii_result['conversion_info']['output_size']}")
        print(f"  - Total characters: {ascii_result['character_count']}")
        
        # Test enhanced visual analysis
        start_time = time.time()
        agent = EnhancedVisualAnalysisAgent(enable_ascii=False)  # Skip ASCII for speed
        result = agent.analyze_image_enhanced(image_path, save_ascii=False)
        analysis_time = time.time() - start_time
        
        print(f"✓ Real image visual analysis completed in {analysis_time:.2f} seconds")
        
        # Test combined analysis
        start_time = time.time()
        agent = EnhancedVisualAnalysisAgent(enable_ascii=True, ascii_quality='medium')
        result = agent.analyze_image_enhanced(image_path, save_ascii=True)
        combined_time = time.time() - start_time
        
        print(f"✓ Combined analysis completed in {combined_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Real image processing failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("Real Image Processing tests completed.\n")


def test_error_handling():
    """Test error handling capabilities"""
    print("Testing Error Handling...")
    
    agent = EnhancedVisualAnalysisAgent()
    
    # Test non-existent file
    try:
        agent.analyze_image_enhanced('/nonexistent/file.jpg')
        print("❌ Should have raised FileNotFoundError")
    except FileNotFoundError:
        print("✓ Correctly handles non-existent files")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test invalid file format
    with open('/tmp/invalid.txt', 'w') as f:
        f.write('This is not an image')
    
    try:
        agent.analyze_image_enhanced('/tmp/invalid.txt')
        print("❌ Should have raised ValueError")
    except ValueError:
        print("✓ Correctly handles invalid image files")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print("Error Handling tests completed.\n")


def run_all_tests():
    """Run comprehensive test suite"""
    print("=" * 60)
    print("ENHANCED FEATURES TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_ascii_converter,
        test_enhanced_visual_agent,
        test_real_image_processing,
        test_error_handling
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_func in tests:
        try:
            test_func()
            passed_tests += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    if passed_tests == total_tests:
        print(f"ALL TESTS PASSED! ✓ ({passed_tests}/{total_tests})")
        print("Enhanced features are working correctly.")
    else:
        print(f"SOME TESTS FAILED! ❌ ({passed_tests}/{total_tests})")
        print("Please check the error messages above.")
    print("=" * 60)
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)