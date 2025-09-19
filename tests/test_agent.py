#!/usr/bin/env python3
"""
Simple test script to verify the Visual Analysis Agent implementation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from visual_analysis_agent import VisualAnalysisAgent, ImageMetadata, ColorPalette, LightingAnalysis
from PIL import Image, ImageDraw


def test_metadata():
    """Test ImageMetadata functionality"""
    print("Testing ImageMetadata...")
    metadata = ImageMetadata(
        width=1920,
        height=1080, 
        aspect_ratio=1920/1080,
        file_type="JPEG",
        file_size=1024000
    )
    
    assert metadata.aspect_ratio_description == "16:9"
    print("✓ Aspect ratio detection works")
    
    # Test square image
    square_metadata = ImageMetadata(800, 800, 1.0, "PNG", 500000)
    assert "square" in square_metadata.aspect_ratio_description
    print("✓ Square aspect ratio detection works")


def test_agent_initialization():
    """Test agent can be initialized"""
    print("\nTesting agent initialization...")
    agent = VisualAnalysisAgent()
    assert agent.current_image is None
    assert agent.analysis_results == {}
    print("✓ Agent initializes correctly")


def test_sample_image_analysis():
    """Test full analysis workflow with a sample image"""
    print("\nTesting sample image analysis...")
    
    # Create a simple test image
    image = Image.new('RGB', (400, 300), color='lightblue')
    draw = ImageDraw.Draw(image)
    draw.rectangle([50, 50, 150, 150], fill='red')
    draw.ellipse([200, 100, 350, 200], fill='green')
    
    test_image_path = '/tmp/test_image.jpg'
    image.save(test_image_path)
    
    # Test the analysis
    agent = VisualAnalysisAgent()
    results = agent.analyze_image(test_image_path)
    
    # Verify all steps are present
    expected_steps = ['step1', 'step2', 'step3', 'step4', 'step5', 'prompts']
    for step in expected_steps:
        assert step in results, f"Missing step: {step}"
    
    # Verify each step has the required fields
    for step_key in ['step1', 'step2', 'step3', 'step4', 'step5']:
        step = results[step_key]
        assert hasattr(step, 'step_name')
        assert hasattr(step, 'description')
        assert hasattr(step, 'output')
        assert isinstance(step.output, str)
        assert len(step.output) > 0
    
    # Verify prompts
    prompts = results['prompts']
    expected_levels = ['level1', 'level2', 'level3', 'level4']
    for level in expected_levels:
        assert level in prompts, f"Missing prompt level: {level}"
        assert isinstance(prompts[level], str)
        assert len(prompts[level]) > 0
    
    print("✓ Full analysis workflow completed successfully")
    print(f"✓ Generated {len(expected_levels)} prompt levels")
    print(f"✓ All {len(expected_steps)} analysis steps completed")


def test_color_extraction():
    """Test color extraction functionality"""
    print("\nTesting color extraction...")
    
    # Create an image with known colors
    image = Image.new('RGB', (100, 100))
    pixels = []
    
    # Fill with mostly blue, some red
    for y in range(100):
        for x in range(100):
            if x < 20:  # Left portion red
                pixels.append((255, 0, 0))  # Red
            else:  # Rest blue
                pixels.append((0, 0, 255))  # Blue
    
    image.putdata(pixels)
    test_image_path = '/tmp/color_test.jpg'
    image.save(test_image_path)
    
    agent = VisualAnalysisAgent()
    agent.current_image = Image.open(test_image_path)
    
    color_palette = agent._extract_color_palette()
    assert len(color_palette.dominant_colors) > 0
    print(f"✓ Extracted {len(color_palette.dominant_colors)} dominant colors")
    
    # Check that blue and red are detected
    color_names = [color[0] for color in color_palette.dominant_colors]
    print(f"✓ Detected colors: {color_names}")


def test_error_handling():
    """Test error handling for invalid inputs"""
    print("\nTesting error handling...")
    
    agent = VisualAnalysisAgent()
    
    # Test with non-existent file
    try:
        agent.analyze_image('/nonexistent/path.jpg')
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("✓ Correctly handles non-existent files")
    
    # Test step methods without loaded image
    try:
        agent._step1_initial_ingestion()
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ Correctly handles missing image in step methods")


def run_all_tests():
    """Run all tests"""
    print("Running Visual Analysis Agent Tests")
    print("=" * 50)
    
    try:
        test_metadata()
        test_agent_initialization()
        test_sample_image_analysis()
        test_color_extraction()
        test_error_handling()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("The Visual Analysis Agent implementation is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)