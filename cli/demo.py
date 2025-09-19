#!/usr/bin/env python3
"""
Demo script for the Visual Analysis Agent

This script demonstrates how to use the VisualAnalysisAgent to analyze images
and generate progressive prompts following the specification in agent.md
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from visual_analysis_agent import VisualAnalysisAgent
from PIL import Image, ImageDraw


def create_sample_image():
    """Create a simple sample image for testing"""
    # Create a 800x600 sample landscape image
    width, height = 800, 600
    image = Image.new('RGB', (width, height), color='lightblue')
    draw = ImageDraw.Draw(image)
    
    # Draw a simple landscape scene
    # Sky gradient (simplified)
    draw.rectangle([0, 0, width, height//3], fill='lightblue')
    
    # Mountains (background)
    mountains = [(0, height//3), (200, height//4), (400, height//3), 
                (600, height//4), (800, height//3), (800, height), (0, height)]
    draw.polygon(mountains, fill='gray')
    
    # Lake (midground)
    draw.ellipse([100, height//2, 700, height*2//3], fill='darkblue')
    
    # Tree (foreground)
    # Tree trunk
    draw.rectangle([350, height//2, 370, height*2//3], fill='brown')
    # Tree crown
    draw.ellipse([320, height//3, 400, height//2 + 20], fill='darkgreen')
    
    # Save the sample image
    sample_path = '/tmp/sample_landscape.jpg'
    image.save(sample_path, 'JPEG')
    return sample_path


def format_analysis_output(results):
    """Format the analysis results for display"""
    print("=" * 60)
    print("VISUAL ANALYSIS RESULTS")
    print("=" * 60)
    
    for step_key in ['step1', 'step2', 'step3', 'step4', 'step5']:
        if step_key in results:
            step = results[step_key]
            print(f"\n{step.step_name.upper()}")
            print("-" * len(step.step_name))
            print(f"Description: {step.description}")
            print(f"Output: {step.output}")
    
    print("\n" + "=" * 60)
    print("PROGRESSIVE PROMPTS")
    print("=" * 60)
    
    prompts = results.get('prompts', {})
    for level in ['level1', 'level2', 'level3', 'level4']:
        if level in prompts:
            level_num = level.replace('level', 'Level ')
            print(f"\n{level_num.upper()}: Core Concept")
            if level == 'level2':
                print(f"{level_num.upper()}: Detailed Composition")
            elif level == 'level3':
                print(f"{level_num.upper()}: Artistic & Atmospheric")
            elif level == 'level4':
                print(f"{level_num.upper()}: Master Blueprint")
            print("-" * 40)
            print(prompts[level])


def demo_with_sample_image():
    """Run the demo with a sample image"""
    print("Creating sample landscape image...")
    sample_path = create_sample_image()
    print(f"Sample image created at: {sample_path}")
    
    print("\nInitializing Visual Analysis Agent...")
    agent = VisualAnalysisAgent()
    
    print("Analyzing image through 5-step workflow...")
    try:
        results = agent.analyze_image(sample_path)
        format_analysis_output(results)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return False
        
    return True


def demo_with_user_image(image_path):
    """Run the demo with a user-provided image"""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False
        
    print(f"Analyzing image: {image_path}")
    agent = VisualAnalysisAgent()
    
    try:
        results = agent.analyze_image(image_path)
        format_analysis_output(results)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return False
        
    return True


def main():
    """Main demo function"""
    print("Visual Analysis Agent Demo")
    print("Based on the specification in agent.md")
    print("-" * 40)
    
    if len(sys.argv) > 1:
        # User provided an image path
        image_path = sys.argv[1]
        success = demo_with_user_image(image_path)
    else:
        # Use sample image
        print("No image path provided. Using sample image.")
        success = demo_with_sample_image()
    
    if success:
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nThe Visual Analysis Agent successfully implemented:")
        print("• 5-step comprehensive image analysis workflow")
        print("• Progressive prompt generation (4 levels)")
        print("• Color palette extraction")
        print("• Compositional analysis")
        print("• Shape and texture assessment")
        print("\nFor production use, enhance the analysis algorithms")
        print("with more sophisticated computer vision techniques.")
    else:
        print("\nDemo failed. Please check the error messages above.")


if __name__ == "__main__":
    main()