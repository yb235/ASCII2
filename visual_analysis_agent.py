"""
Visual Analysis Agent - Implementation of the Visual Deconstruction & Prompt Synthesis Agent

This module implements a comprehensive visual analysis system that can deconstruct images
into their fundamental components and generate progressive text prompts for image generation.

Based on the agent specification in agent.md
"""

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import colorsys
import numpy as np
from collections import Counter


@dataclass
class ImageMetadata:
    """Basic image properties and metadata"""
    width: int
    height: int
    aspect_ratio: float
    file_type: str
    file_size: int
    
    @property
    def aspect_ratio_description(self) -> str:
        """Get a human-readable aspect ratio description"""
        if abs(self.aspect_ratio - 16/9) < 0.1:
            return "16:9"
        elif abs(self.aspect_ratio - 4/3) < 0.1:
            return "4:3"
        elif abs(self.aspect_ratio - 1.0) < 0.1:
            return "1:1 (square)"
        elif abs(self.aspect_ratio - 3/2) < 0.1:
            return "3:2"
        else:
            return f"{self.aspect_ratio:.2f}:1"


@dataclass
class ColorPalette:
    """Color analysis results"""
    dominant_colors: List[Tuple[str, str]]  # (color_name, hex_code)
    accent_colors: List[Tuple[str, str]]
    color_harmony: str
    color_temperature: str  # warm, cool, neutral


@dataclass
class LightingAnalysis:
    """Lighting analysis results"""
    light_source: str
    light_direction: str
    light_quality: str  # soft, hard, diffused
    shadow_description: str
    highlight_description: str


@dataclass
class AnalysisStep:
    """Base class for analysis step results"""
    step_name: str
    description: str
    output: str


class VisualAnalysisAgent:
    """
    Main visual analysis agent that implements the 5-step workflow
    described in the agent specification.
    """
    
    def __init__(self):
        """Initialize the visual analysis agent"""
        self.current_image = None
        self.analysis_results = {}
        
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Main analysis method that processes an image through all 5 steps
        
        Args:
            image_path: Path to the image file to analyze
            
        Returns:
            Complete analysis results dictionary
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Load the image
        self.current_image = Image.open(image_path)
        
        # Execute the 5-step workflow
        results = {}
        results['step1'] = self._step1_initial_ingestion()
        results['step2'] = self._step2_compositional_analysis()
        results['step3'] = self._step3_shape_deconstruction()
        results['step4'] = self._step4_color_lighting_analysis()
        results['step5'] = self._step5_texture_material_analysis()
        
        # Generate progressive prompts
        results['prompts'] = self._generate_prompts(results)
        
        self.analysis_results = results
        return results
    
    def _step1_initial_ingestion(self) -> AnalysisStep:
        """
        Step 1: Initial Ingestion & High-Level Triage
        Identify basic properties and categorize the image
        """
        if not self.current_image:
            raise ValueError("No image loaded")
            
        # Extract metadata
        width, height = self.current_image.size
        aspect_ratio = width / height
        file_format = self.current_image.format or "Unknown"
        
        metadata = ImageMetadata(
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            file_type=file_format,
            file_size=0  # Would need actual file size from path
        )
        
        # Classify image category (simplified heuristic)
        category = self._classify_image_category()
        
        # Assess initial mood (simplified heuristic)
        mood = self._assess_initial_mood()
        
        description = f"Image analysis of {width}x{height} {file_format} image"
        output = f"{metadata.aspect_ratio_description} aspect ratio {category.lower()}. {mood} mood."
        
        return AnalysisStep(
            step_name="Initial Ingestion & High-Level Triage",
            description=description,
            output=output
        )
    
    def _step2_compositional_analysis(self) -> AnalysisStep:
        """
        Step 2: Compositional & Structural Analysis
        Analyze arrangement of elements and overall structure
        """
        # Simplified compositional analysis
        layout_rule = self._determine_layout_rule()
        focal_points = self._identify_focal_points()
        depth_layers = self._analyze_depth_layers()
        negative_space = self._analyze_negative_space()
        
        description = "Analysis of compositional structure and element arrangement"
        output = f"Composition follows {layout_rule}. {focal_points} {depth_layers} {negative_space}"
        
        return AnalysisStep(
            step_name="Compositional & Structural Analysis",
            description=description,
            output=output
        )
    
    def _step3_shape_deconstruction(self) -> AnalysisStep:
        """
        Step 3: Shape & Form Deconstruction
        Break down objects into geometric and organic shapes
        """
        # Simplified shape analysis
        geometric_shapes = self._analyze_geometric_shapes()
        organic_shapes = self._analyze_organic_shapes()
        
        description = "Deconstruction of objects into constituent shapes"
        output = f"{geometric_shapes} {organic_shapes}"
        
        return AnalysisStep(
            step_name="Shape & Form Deconstruction",
            description=description,
            output=output
        )
    
    def _step4_color_lighting_analysis(self) -> AnalysisStep:
        """
        Step 4: Color Palette & Lighting Analysis
        Detailed analysis of colors and lighting
        """
        # Extract dominant colors
        color_palette = self._extract_color_palette()
        lighting_analysis = self._analyze_lighting()
        
        description = "Forensic analysis of color palette and lighting conditions"
        color_desc = f"Color palette dominated by {', '.join([c[0] for c in color_palette.dominant_colors[:3]])}."
        lighting_desc = f"Lighting: {lighting_analysis.light_quality} {lighting_analysis.light_source} from {lighting_analysis.light_direction}."
        
        output = f"{color_desc} {lighting_desc}"
        
        return AnalysisStep(
            step_name="Color Palette & Lighting Analysis", 
            description=description,
            output=output
        )
    
    def _step5_texture_material_analysis(self) -> AnalysisStep:
        """
        Step 5: Texture & Material Definition
        Analyze surface qualities and materials
        """
        # Simplified texture analysis
        textures = self._analyze_textures()
        patterns = self._identify_patterns()
        
        description = "Analysis of surface textures and material properties"
        output = f"Textures: {textures}. Patterns: {patterns}"
        
        return AnalysisStep(
            step_name="Texture & Material Definition",
            description=description,
            output=output
        )
    
    def _generate_prompts(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate the 4-level progressive prompts based on analysis results
        """
        # Extract key information from analysis results
        step1 = results['step1']
        step2 = results['step2'] 
        step3 = results['step3']
        step4 = results['step4']
        step5 = results['step5']
        
        # Level 1: Core Concept
        level1 = self._generate_level1_prompt()
        
        # Level 2: Detailed Composition  
        level2 = f"{level1}, {step2.output.lower()}"
        
        # Level 3: Artistic & Atmospheric
        level3 = f"{level2}. {step4.output}"
        
        # Level 4: Master Blueprint
        level4 = f"{level3} {step5.output} {step3.output}"
        
        return {
            "level1": level1,
            "level2": level2, 
            "level3": level3,
            "level4": level4
        }
    
    # Helper methods for analysis steps
    
    def _classify_image_category(self) -> str:
        """Classify image into high-level category (simplified)"""
        # This would use more sophisticated image analysis in a real implementation
        categories = ["Portrait", "Landscape", "Architectural", "Abstract", "Still Life", "Concept Art"]
        return "General image"  # Simplified for now
    
    def _assess_initial_mood(self) -> str:
        """Assess initial mood of the image (simplified)"""
        moods = ["Serene", "Chaotic", "Joyful", "Melancholy", "Futuristic", "Peaceful"]
        return "Neutral"  # Simplified for now
    
    def _determine_layout_rule(self) -> str:
        """Determine compositional layout rule (simplified)"""
        rules = ["Rule of Thirds", "Golden Ratio", "Centered", "Symmetrical", "Asymmetrical Balance"]
        return "balanced composition"  # Simplified
    
    def _identify_focal_points(self) -> str:
        """Identify focal points in the image (simplified)"""
        return "Central focal point"  # Simplified
    
    def _analyze_depth_layers(self) -> str:
        """Analyze foreground, midground, background (simplified)"""
        return "Clear depth with foreground, midground, and background elements."
    
    def _analyze_negative_space(self) -> str:
        """Analyze negative space usage (simplified)"""
        return "Negative space provides balance."
    
    def _analyze_geometric_shapes(self) -> str:
        """Analyze geometric shapes (simplified)"""
        return "Geometric elements include rectangular and circular forms."
    
    def _analyze_organic_shapes(self) -> str:
        """Analyze organic shapes (simplified)"""
        return "Organic shapes with natural, flowing lines."
    
    def _extract_color_palette(self) -> ColorPalette:
        """Extract dominant colors from the image"""
        # Convert image to RGB if necessary
        if self.current_image.mode != 'RGB':
            rgb_image = self.current_image.convert('RGB')
        else:
            rgb_image = self.current_image
            
        # Get color data
        colors = rgb_image.getcolors(maxcolors=256*256*256)
        if colors is None:
            # Image has too many colors, resize and try again
            small_image = rgb_image.resize((150, 150))
            colors = small_image.getcolors(maxcolors=256*256*256)
        
        # Sort by frequency and get top colors
        sorted_colors = sorted(colors, key=lambda x: x[0], reverse=True)
        
        # Convert to hex and name (simplified naming)
        dominant_colors = []
        for i, (count, rgb) in enumerate(sorted_colors[:5]):
            hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            color_name = self._get_color_name(rgb)
            dominant_colors.append((color_name, hex_color))
        
        return ColorPalette(
            dominant_colors=dominant_colors,
            accent_colors=[],  # Simplified
            color_harmony="analogous",  # Simplified
            color_temperature="neutral"  # Simplified
        )
    
    def _get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Get a simple color name from RGB values"""
        r, g, b = rgb
        
        # Simple color classification
        if r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif r > g and r > b:
            return "red"
        elif g > r and g > b:
            return "green"
        elif b > r and b > g:
            return "blue"
        elif r > 150 and g > 150:
            return "yellow"
        elif r > 150 and b > 150:
            return "magenta"
        elif g > 150 and b > 150:
            return "cyan"
        else:
            return "gray"
    
    def _analyze_lighting(self) -> LightingAnalysis:
        """Analyze lighting conditions (simplified)"""
        return LightingAnalysis(
            light_source="natural light",
            light_direction="from above",
            light_quality="soft",
            shadow_description="gentle shadows",
            highlight_description="subtle highlights"
        )
    
    def _analyze_textures(self) -> str:
        """Analyze surface textures (simplified)"""
        return "smooth and varied surface textures"
    
    def _identify_patterns(self) -> str:
        """Identify repeating patterns (simplified)"""
        return "no obvious repeating patterns"
    
    def _generate_level1_prompt(self) -> str:
        """Generate Level 1 core concept prompt"""
        return "A detailed image"  # Would be more sophisticated based on actual analysis


def main():
    """Example usage of the Visual Analysis Agent"""
    agent = VisualAnalysisAgent()
    
    # This would work with actual image files
    print("Visual Analysis Agent initialized.")
    print("Usage: agent.analyze_image('path/to/image.jpg')")
    print("\nThe agent implements the 5-step workflow:")
    print("1. Initial Ingestion & High-Level Triage")
    print("2. Compositional & Structural Analysis")
    print("3. Shape & Form Deconstruction") 
    print("4. Color Palette & Lighting Analysis")
    print("5. Texture & Material Definition")
    print("\nAnd generates 4 levels of progressive prompts for image generation.")


if __name__ == "__main__":
    main()