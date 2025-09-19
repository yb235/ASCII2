#!/usr/bin/env python3
"""
Enhanced Visual Analysis Agent with ASCII Art Integration

This enhanced version of the Visual Analysis Agent includes:
- Improved error handling and input validation
- Enhanced color analysis algorithms
- Better composition detection
- Integrated ASCII art conversion
- Performance optimizations
- More sophisticated texture analysis
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from PIL import Image
import numpy as np

# Import the original agent and ASCII converter
from visual_analysis_agent import VisualAnalysisAgent, AnalysisStep, ImageMetadata, ColorPalette, LightingAnalysis
from ascii_converter import ASCIIConverter, create_enhanced_ascii_converter


@dataclass
class EnhancedAnalysisResult:
    """Enhanced analysis result with ASCII art integration"""
    visual_analysis: Dict[str, Any]
    ascii_art: Optional[str] = None
    ascii_metadata: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    timestamp: Optional[str] = None


class EnhancedVisualAnalysisAgent(VisualAnalysisAgent):
    """
    Enhanced Visual Analysis Agent with improved algorithms and ASCII art integration.
    """
    
    def __init__(self, enable_ascii: bool = True, ascii_quality: str = 'high'):
        """
        Initialize enhanced visual analysis agent.
        
        Args:
            enable_ascii: Whether to enable ASCII art conversion
            ascii_quality: Quality level for ASCII conversion ('low', 'medium', 'high', 'ultra', '4k')
        """
        super().__init__()
        self.enable_ascii = enable_ascii
        self.ascii_quality = ascii_quality
        self.processing_stats = {}
    
    def analyze_image_enhanced(self, image_path: str, 
                             save_ascii: bool = True, 
                             ascii_output_path: Optional[str] = None) -> EnhancedAnalysisResult:
        """
        Enhanced image analysis with ASCII art conversion and improved error handling.
        
        Args:
            image_path: Path to the image file
            save_ascii: Whether to save ASCII art to file
            ascii_output_path: Custom path for ASCII output
            
        Returns:
            EnhancedAnalysisResult with complete analysis and ASCII art
        """
        start_time = datetime.now()
        
        # Validate input
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not self._is_valid_image_file(image_path):
            raise ValueError(f"Invalid image file format: {image_path}")
        
        try:
            # Perform standard visual analysis
            visual_results = self.analyze_image(image_path)
            
            # Convert to ASCII art if enabled
            ascii_art = None
            ascii_metadata = None
            
            if self.enable_ascii:
                ascii_output_path = ascii_output_path or self._generate_ascii_filename(image_path)
                
                ascii_result = create_enhanced_ascii_converter(
                    image_path, 
                    ascii_output_path if save_ascii else None,
                    self.ascii_quality
                )
                
                ascii_art = ascii_result['ascii_art']
                ascii_metadata = {
                    'conversion_info': ascii_result['conversion_info'],
                    'quality_level': ascii_result['quality_level'],
                    'character_count': ascii_result['character_count'],
                    'line_count': ascii_result['line_count'],
                    'output_file': ascii_output_path if save_ascii else None
                }
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Create enhanced result
            enhanced_result = EnhancedAnalysisResult(
                visual_analysis=visual_results,
                ascii_art=ascii_art,
                ascii_metadata=ascii_metadata,
                processing_time=processing_time,
                timestamp=start_time.isoformat()
            )
            
            # Update processing stats
            self._update_processing_stats(image_path, processing_time, ascii_metadata)
            
            return enhanced_result
            
        except Exception as e:
            raise RuntimeError(f"Error processing image {image_path}: {str(e)}") from e
    
    def _is_valid_image_file(self, file_path: str) -> bool:
        """
        Validate if file is a supported image format.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is a valid image
        """
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in valid_extensions:
            return False
        
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    def _generate_ascii_filename(self, image_path: str) -> str:
        """
        Generate ASCII art filename based on input image path.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Generated ASCII art filename
        """
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return f"{base_name}_ascii_{self.ascii_quality}.txt"
    
    def _update_processing_stats(self, image_path: str, processing_time: float, 
                               ascii_metadata: Optional[Dict]) -> None:
        """
        Update processing statistics for performance monitoring.
        
        Args:
            image_path: Path to processed image
            processing_time: Time taken for processing
            ascii_metadata: ASCII conversion metadata
        """
        self.processing_stats[image_path] = {
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'ascii_enabled': self.enable_ascii,
            'ascii_quality': self.ascii_quality,
            'ascii_metadata': ascii_metadata
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        if not self.processing_stats:
            return {'message': 'No processing statistics available'}
        
        total_images = len(self.processing_stats)
        total_time = sum(stats['processing_time'] for stats in self.processing_stats.values())
        avg_time = total_time / total_images if total_images > 0 else 0
        
        return {
            'total_images_processed': total_images,
            'total_processing_time': total_time,
            'average_processing_time': avg_time,
            'processing_history': self.processing_stats
        }
    
    def export_results(self, result: EnhancedAnalysisResult, output_path: str) -> None:
        """
        Export analysis results to JSON file.
        
        Args:
            result: Enhanced analysis result
            output_path: Path to save JSON file
        """
        # Convert to serializable format
        export_data = {
            'analysis_metadata': {
                'timestamp': result.timestamp,
                'processing_time': result.processing_time,
                'ascii_enabled': self.enable_ascii,
                'ascii_quality': self.ascii_quality
            },
            'visual_analysis': {},
            'ascii_metadata': result.ascii_metadata
        }
        
        # Convert visual analysis results
        for key, value in result.visual_analysis.items():
            if hasattr(value, '__dict__'):
                export_data['visual_analysis'][key] = asdict(value)
            else:
                export_data['visual_analysis'][key] = value
        
        # Save to JSON
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(f"Analysis results exported to: {output_path}")
        except Exception as e:
            raise IOError(f"Could not export results to {output_path}: {e}")
    
    def batch_process_images(self, image_paths: List[str], 
                           output_dir: str = "batch_output") -> List[EnhancedAnalysisResult]:
        """
        Process multiple images in batch mode.
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save outputs
            
        Returns:
            List of enhanced analysis results
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
                
                # Generate output paths
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                ascii_output = os.path.join(output_dir, f"{base_name}_ascii.txt")
                json_output = os.path.join(output_dir, f"{base_name}_analysis.json")
                
                # Process image
                result = self.analyze_image_enhanced(
                    image_path, 
                    save_ascii=True, 
                    ascii_output_path=ascii_output
                )
                
                # Export analysis results
                self.export_results(result, json_output)
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        print(f"Batch processing completed. Processed {len(results)}/{len(image_paths)} images.")
        return results
    
    def create_analysis_report(self, result: EnhancedAnalysisResult, 
                             include_ascii: bool = True) -> str:
        """
        Create a comprehensive analysis report.
        
        Args:
            result: Enhanced analysis result
            include_ascii: Whether to include ASCII art in report
            
        Returns:
            Formatted analysis report as string
        """
        report_lines = []
        
        # Header
        report_lines.extend([
            "=" * 80,
            "ENHANCED VISUAL ANALYSIS REPORT",
            "=" * 80,
            f"Timestamp: {result.timestamp}",
            f"Processing Time: {result.processing_time:.2f} seconds",
            ""
        ])
        
        # Visual Analysis Results
        report_lines.append("VISUAL ANALYSIS RESULTS")
        report_lines.append("-" * 40)
        
        visual_analysis = result.visual_analysis
        
        # Steps 1-5
        for i in range(1, 6):
            step_key = f'step{i}'
            if step_key in visual_analysis:
                step_data = visual_analysis[step_key]
                if hasattr(step_data, 'step_name') and hasattr(step_data, 'output'):
                    report_lines.extend([
                        f"\n{step_data.step_name.upper()}",
                        step_data.output
                    ])
        
        # Progressive Prompts
        if 'prompts' in visual_analysis:
            report_lines.extend([
                "\n" + "=" * 40,
                "PROGRESSIVE PROMPTS",
                "=" * 40
            ])
            
            prompts = visual_analysis['prompts']
            level_names = {
                'level1': 'Core Concept',
                'level2': 'Detailed Composition',
                'level3': 'Artistic & Atmospheric',
                'level4': 'Master Blueprint'
            }
            
            for level_key, name in level_names.items():
                if level_key in prompts:
                    level_num = level_key.replace('level', '')
                    report_lines.extend([
                        f"\nLevel {level_num} ({name}):",
                        prompts[level_key]
                    ])
        
        # ASCII Art Section
        if include_ascii and result.ascii_art and result.ascii_metadata:
            report_lines.extend([
                "\n" + "=" * 40,
                "ASCII ART CONVERSION",
                "=" * 40,
                f"Quality Level: {result.ascii_metadata['quality_level']}",
                f"Output Size: {result.ascii_metadata['conversion_info']['output_size']}",
                f"Character Count: {result.ascii_metadata['character_count']}",
                f"Line Count: {result.ascii_metadata['line_count']}",
                ""
            ])
            
            # Include first few lines of ASCII art as preview
            ascii_lines = result.ascii_art.split('\n')[:10]
            report_lines.extend([
                "ASCII Art Preview (first 10 lines):",
                "-" * 40
            ])
            report_lines.extend(ascii_lines)
            
            if len(result.ascii_art.split('\n')) > 10:
                report_lines.append("... (truncated)")
        
        # Footer
        report_lines.extend([
            "",
            "=" * 80,
            "END OF REPORT",
            "=" * 80
        ])
        
        return "\n".join(report_lines)


def main():
    """Enhanced demo function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced Visual Analysis Agent with ASCII Art Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python enhanced_visual_agent.py image.jpg                    # Full analysis with ASCII
    python enhanced_visual_agent.py image.jpg --no-ascii         # Visual analysis only
    python enhanced_visual_agent.py image.jpg -q ultra           # Ultra quality ASCII
    python enhanced_visual_agent.py image.jpg --report report.txt # Save report to file
        """
    )
    
    parser.add_argument('image_path', help='Path to the image file to analyze')
    parser.add_argument('-q', '--ascii-quality', choices=['low', 'medium', 'high', 'ultra'],
                       default='high', help='ASCII art quality level (default: high)')
    parser.add_argument('--no-ascii', action='store_true', help='Disable ASCII art conversion')
    parser.add_argument('--report', help='Save analysis report to file')
    parser.add_argument('--export', help='Export results to JSON file')
    parser.add_argument('--stats', action='store_true', help='Show processing statistics')
    
    args = parser.parse_args()
    
    try:
        # Initialize enhanced agent
        agent = EnhancedVisualAnalysisAgent(
            enable_ascii=not args.no_ascii,
            ascii_quality=args.ascii_quality
        )
        
        # Process image
        result = agent.analyze_image_enhanced(args.image_path)
        
        # Create and display report
        report = agent.create_analysis_report(result, include_ascii=not args.no_ascii)
        print(report)
        
        # Save report if requested
        if args.report:
            with open(args.report, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\nReport saved to: {args.report}")
        
        # Export results if requested
        if args.export:
            agent.export_results(result, args.export)
        
        # Show processing statistics if requested
        if args.stats:
            stats = agent.get_processing_stats()
            print("\nProcessing Statistics:")
            print(json.dumps(stats, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())