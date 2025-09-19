#!/usr/bin/env python3
"""
Command Line Interface for the Visual Analysis Agent

Usage:
    python cli.py <image_path> [options]
    
Options:
    --level     Show specific prompt level (1-4)
    --step      Show specific analysis step (1-5)
    --json      Output results in JSON format
    --verbose   Show detailed analysis
"""

import argparse
import json
import sys
import os
from visual_analysis_agent import VisualAnalysisAgent


def format_step_output(step_data, verbose=False):
    """Format step output for display"""
    if verbose:
        return f"{step_data.step_name}:\n{step_data.description}\n\nOutput: {step_data.output}\n"
    else:
        return f"{step_data.step_name}: {step_data.output}"


def main():
    parser = argparse.ArgumentParser(
        description="Visual Analysis Agent - Analyze images and generate prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cli.py image.jpg                    # Full analysis
    python cli.py image.jpg --level 4          # Show only Level 4 prompt
    python cli.py image.jpg --step 1           # Show only Step 1 analysis
    python cli.py image.jpg --json             # JSON output
    python cli.py image.jpg --verbose          # Detailed output
        """
    )
    
    parser.add_argument('image_path', help='Path to the image file to analyze')
    parser.add_argument('--level', type=int, choices=[1, 2, 3, 4], 
                       help='Show specific prompt level (1-4)')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4, 5],
                       help='Show specific analysis step (1-5)')
    parser.add_argument('--json', action='store_true',
                       help='Output results in JSON format')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed analysis information')
    
    args = parser.parse_args()
    
    # Check if image file exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize and run analysis
    try:
        agent = VisualAnalysisAgent()
        results = agent.analyze_image(args.image_path)
        
        if args.json:
            # Convert results to JSON-serializable format
            json_results = {}
            for key, value in results.items():
                if key == 'prompts':
                    json_results[key] = value
                else:
                    json_results[key] = {
                        'step_name': value.step_name,
                        'description': value.description,
                        'output': value.output
                    }
            
            print(json.dumps(json_results, indent=2))
            
        elif args.level:
            # Show specific prompt level
            level_key = f'level{args.level}'
            if level_key in results['prompts']:
                print(f"Level {args.level} Prompt:")
                print(results['prompts'][level_key])
            else:
                print(f"Error: Level {args.level} not found", file=sys.stderr)
                sys.exit(1)
                
        elif args.step:
            # Show specific analysis step
            step_key = f'step{args.step}'
            if step_key in results:
                print(format_step_output(results[step_key], args.verbose))
            else:
                print(f"Error: Step {args.step} not found", file=sys.stderr)
                sys.exit(1)
                
        else:
            # Show full analysis
            print(f"Visual Analysis Results for: {args.image_path}")
            print("=" * 60)
            
            # Show all analysis steps
            for i in range(1, 6):
                step_key = f'step{i}'
                if step_key in results:
                    print(f"\n{format_step_output(results[step_key], args.verbose)}")
            
            # Show all prompt levels
            print(f"\n{'='*60}")
            print("Generated Prompts:")
            print("=" * 60)
            
            prompts = results['prompts']
            level_names = {
                'level1': 'Core Concept',
                'level2': 'Detailed Composition', 
                'level3': 'Artistic & Atmospheric',
                'level4': 'Master Blueprint'
            }
            
            for level_key, name in level_names.items():
                if level_key in prompts:
                    level_num = level_key.replace('level', '')
                    print(f"\nLevel {level_num} ({name}):")
                    print(f"{prompts[level_key]}")
                    
    except Exception as e:
        print(f"Error analyzing image: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()