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
        Following the exact specification formulas from agent.md
        """
        # Extract key information from analysis results
        step1 = results['step1']
        step2 = results['step2'] 
        step3 = results['step3']
        step4 = results['step4']
        step5 = results['step5']
        
        # Level 1: Core Concept - [Style] of [Primary Subject] in a [Setting]
        level1 = self._generate_level1_prompt()
        
        # Level 2: Detailed Composition - [L1 Prompt] + [Composition] + [Key Elements & Placement]
        composition_details = step2.output.replace("Composition follows ", "").replace(". ", ", ")
        level2 = f"{level1}, {composition_details.lower()}"
        
        # Level 3: Artistic & Atmospheric - [L2 Prompt] + [Lighting Description] + [Color Palette] + [Mood]
        # Extract lighting and color from step4
        step4_parts = step4.output.split(". ")
        color_info = step4_parts[0] if len(step4_parts) > 0 else "with natural colors"
        lighting_info = step4_parts[1] if len(step4_parts) > 1 else "with ambient lighting"
        
        mood = self._assess_initial_mood().lower()
        level3 = f"{level2}. {lighting_info}. {color_info}, creating a {mood} mood"
        
        # Level 4: Master Blueprint - [L3 Prompt] + [Textural Details] + [Shape Language] + [Advanced Modifiers]
        texture_info = step5.output.replace("Textures: ", "").replace(". Patterns: ", ", patterns: ")
        shape_info = step3.output
        
        # Add technical photography details for Level 4
        technical_details = "Shot with sharp focus, high dynamic range, professional lighting"
        
        level4 = f"{level3}. {texture_info}. {shape_info}. {technical_details}"
        
        return {
            "level1": level1,
            "level2": level2, 
            "level3": level3,
            "level4": level4
        }
    
    # Helper methods for analysis steps
    
    def _classify_image_category(self) -> str:
        """Classify image into high-level category with basic heuristics"""
        if not self.current_image:
            return "General image"
            
        # Basic heuristics based on image properties
        width, height = self.current_image.size
        aspect_ratio = width / height
        
        # Analyze color distribution to help classify
        color_palette = self._extract_color_palette()
        dominant_colors = [color[0] for color in color_palette.dominant_colors[:3]]
        
        # Portrait detection - typically vertical orientation with specific aspect ratios
        if aspect_ratio < 0.8:  # Taller than wide
            return "Portrait"
        
        # Landscape detection - typically horizontal with natural colors
        elif aspect_ratio > 1.5 and any(color in dominant_colors for color in ['green', 'blue', 'gray']):
            return "Landscape"
        
        # Architectural detection - often geometric shapes and neutral colors
        elif any(color in dominant_colors for color in ['gray', 'white', 'black']):
            return "Architectural"
        
        # Abstract detection - based on color diversity
        elif len(set(dominant_colors)) >= 4:
            return "Abstract"
        
        # Still life - typically square or slightly rectangular
        elif 0.8 <= aspect_ratio <= 1.5:
            return "Still Life"
        
        else:
            return "General image"
    
    def _assess_initial_mood(self) -> str:
        """Assess initial mood based on color and brightness analysis"""
        if not self.current_image:
            return "Neutral"
            
        # Convert to numpy array for analysis
        img_array = np.array(self.current_image.convert('RGB'))
        
        # Calculate overall brightness
        brightness = np.mean(img_array)
        
        # Calculate color variance (higher variance = more chaotic)
        color_variance = np.var(img_array)
        
        # Analyze color distribution
        color_palette = self._extract_color_palette()
        dominant_colors = [color[0] for color in color_palette.dominant_colors[:3]]
        
        # Mood classification based on brightness and colors
        if brightness > 180:
            if 'yellow' in dominant_colors or 'white' in dominant_colors:
                return "Joyful"
            else:
                return "Serene"
        elif brightness < 80:
            if 'blue' in dominant_colors or 'gray' in dominant_colors:
                return "Melancholy"
            else:
                return "Mysterious"
        elif color_variance > 3000:  # High color variation
            return "Chaotic"
        elif 'blue' in dominant_colors and 'green' in dominant_colors:
            return "Peaceful"
        elif 'red' in dominant_colors:
            return "Dramatic"
        else:
            return "Neutral"
    
    def _determine_layout_rule(self) -> str:
        """Determine compositional layout rule using basic image analysis"""
        if not self.current_image:
            return "balanced composition"
            
        width, height = self.current_image.size
        aspect_ratio = width / height
        
        # Convert to grayscale for analysis
        gray_image = self.current_image.convert('L')
        img_array = np.array(gray_image)
        
        # Analyze brightness distribution to detect composition patterns
        
        # Rule of Thirds detection - check if there are intensity changes at 1/3 points
        third_x = width // 3
        two_third_x = 2 * width // 3
        third_y = height // 3
        two_third_y = 2 * height // 3
        
        # Sample lines at thirds
        vertical_variance_1 = np.var(img_array[:, third_x])
        vertical_variance_2 = np.var(img_array[:, two_third_x])
        horizontal_variance_1 = np.var(img_array[third_y, :])
        horizontal_variance_2 = np.var(img_array[two_third_y, :])
        
        # If high variance at thirds, likely rule of thirds
        avg_variance = (vertical_variance_1 + vertical_variance_2 + horizontal_variance_1 + horizontal_variance_2) / 4
        if avg_variance > 1000:
            return "rule of thirds"
        
        # Centered composition - check if center has significantly different brightness
        center_x, center_y = width // 2, height // 2
        center_region = img_array[center_y-20:center_y+20, center_x-20:center_x+20]
        edge_brightness = (np.mean(img_array[:20, :]) + np.mean(img_array[-20:, :]) + 
                          np.mean(img_array[:, :20]) + np.mean(img_array[:, -20:])) / 4
        center_brightness = np.mean(center_region)
        
        if abs(center_brightness - edge_brightness) > 30:
            return "centered composition"
        
        # Symmetry detection - compare left/right and top/bottom
        left_half = img_array[:, :width//2]
        right_half = np.fliplr(img_array[:, width//2:])
        symmetry_score = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
        
        if symmetry_score > 0.8:
            return "symmetrical composition"
        
        # Golden ratio approximation (1.618)
        if abs(aspect_ratio - 1.618) < 0.1:
            return "golden ratio composition"
        
        # Default cases
        if aspect_ratio > 1.5:
            return "horizontal balance"
        elif aspect_ratio < 0.8:
            return "vertical emphasis"
        else:
            return "balanced composition"
    
    def _identify_focal_points(self) -> str:
        """Identify focal points using contrast and position analysis"""
        if not self.current_image:
            return "Central focal point"
            
        width, height = self.current_image.size
        gray_image = self.current_image.convert('L')
        img_array = np.array(gray_image)
        
        # Find areas of high contrast (potential focal points)
        # Use a simple edge detection approach
        center_x, center_y = width // 2, height // 2
        
        # Divide image into 9 regions (rule of thirds grid)
        third_x = width // 3
        two_third_x = 2 * width // 3
        third_y = height // 3
        two_third_y = 2 * height // 3
        
        regions = {
            "upper-left": img_array[:third_y, :third_x],
            "upper-center": img_array[:third_y, third_x:two_third_x],
            "upper-right": img_array[:third_y, two_third_x:],
            "middle-left": img_array[third_y:two_third_y, :third_x],
            "center": img_array[third_y:two_third_y, third_x:two_third_x],
            "middle-right": img_array[third_y:two_third_y, two_third_x:],
            "lower-left": img_array[two_third_y:, :third_x],
            "lower-center": img_array[two_third_y:, third_x:two_third_x],
            "lower-right": img_array[two_third_y:, two_third_x:]
        }
        
        # Calculate variance for each region (higher variance = more activity/interest)
        region_variance = {}
        for name, region in regions.items():
            if region.size > 0:
                region_variance[name] = np.var(region)
        
        # Find the region with highest variance
        if region_variance:
            focal_region = max(region_variance, key=region_variance.get)
            max_variance = region_variance[focal_region]
            
            # If center has high variance, it's likely centered
            if focal_region == "center" and max_variance > 1000:
                return "Strong central focal point"
            elif "upper" in focal_region:
                return f"Primary focal point in the {focal_region} region"
            elif "lower" in focal_region:
                return f"Focal point positioned in the {focal_region} area"
            elif "left" in focal_region or "right" in focal_region:
                return f"Focal point located in the {focal_region} section"
            else:
                return f"Focal point in the {focal_region} region"
        
        return "Distributed focal points throughout the composition"
    
    def _analyze_depth_layers(self) -> str:
        """Analyze foreground, midground, background using blur and focus analysis"""
        if not self.current_image:
            return "Clear depth with foreground, midground, and background elements."
            
        # Convert to grayscale for depth analysis
        gray_image = self.current_image.convert('L')
        img_array = np.array(gray_image)
        height, width = img_array.shape
        
        # Divide image into horizontal bands to analyze depth
        # Typically: top = background, middle = midground, bottom = foreground
        top_third = img_array[:height//3, :]
        middle_third = img_array[height//3:2*height//3, :]
        bottom_third = img_array[2*height//3:, :]
        
        layers = []
        
        try:
            # Analyze sharpness/detail in each layer using local variance
            def calculate_sharpness(region):
                if region.size == 0:
                    return 0
                # Use Laplacian-like operator for sharpness
                kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
                # Simple convolution approximation
                edges = np.abs(np.gradient(region.flatten())).mean()
                return edges
            
            top_sharpness = calculate_sharpness(top_third)
            middle_sharpness = calculate_sharpness(middle_third)
            bottom_sharpness = calculate_sharpness(bottom_third)
            
            # Analyze brightness (often background is brighter/hazier)
            top_brightness = np.mean(top_third)
            middle_brightness = np.mean(middle_third)
            bottom_brightness = np.mean(bottom_third)
            
            # Determine layer characteristics
            sharpness_values = [top_sharpness, middle_sharpness, bottom_sharpness]
            max_sharpness = max(sharpness_values)
            
            # Background analysis (top third)
            if top_brightness > middle_brightness + 10:
                layers.append("bright background")
            elif top_sharpness < max_sharpness * 0.7:
                layers.append("soft background")
            else:
                layers.append("detailed background")
            
            # Midground analysis (middle third)
            if middle_sharpness > max_sharpness * 0.8:
                layers.append("sharp midground")
            else:
                layers.append("transitional midground")
            
            # Foreground analysis (bottom third)
            if bottom_sharpness == max_sharpness:
                layers.append("prominent foreground")
            elif bottom_brightness < middle_brightness - 10:
                layers.append("shadowed foreground")
            else:
                layers.append("clear foreground")
                
            # Analyze depth separation
            brightness_range = max(top_brightness, middle_brightness, bottom_brightness) - min(top_brightness, middle_brightness, bottom_brightness)
            if brightness_range > 50:
                depth_quality = "Strong depth separation"
            elif brightness_range > 20:
                depth_quality = "Moderate depth separation"
            else:
                depth_quality = "Subtle depth layers"
                
        except Exception:
            # Fallback analysis
            layers = ["background elements", "midground features", "foreground subjects"]
            depth_quality = "Clear depth"
        
        return f"{depth_quality} with {', '.join(layers)}"
    
    def _analyze_negative_space(self) -> str:
        """Analyze negative space usage (simplified)"""
        return "Negative space provides balance."
    
    def _analyze_geometric_shapes(self) -> str:
        """Analyze geometric shapes using edge detection and contour analysis"""
        if not self.current_image:
            return "Geometric elements include rectangular and circular forms"
            
        # Convert to grayscale for shape analysis
        gray_image = self.current_image.convert('L')
        img_array = np.array(gray_image)
        
        shapes_found = []
        
        # Simple edge detection using gradients
        try:
            gradient_x = np.gradient(img_array, axis=1)
            gradient_y = np.gradient(img_array, axis=0)
            edge_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            
            # Threshold to find strong edges
            edge_threshold = np.mean(edge_magnitude) + 2 * np.std(edge_magnitude)
            strong_edges = edge_magnitude > edge_threshold
            
            # Count edge density to infer shape complexity
            edge_density = np.sum(strong_edges) / strong_edges.size
            
            if edge_density > 0.1:
                shapes_found.append("complex geometric forms")
            elif edge_density > 0.05:
                shapes_found.append("moderate geometric detail")
            else:
                shapes_found.append("simple geometric forms")
                
            # Analyze edge patterns to infer shape types
            # Look for straight lines (rectangular shapes)
            horizontal_edges = np.sum(strong_edges, axis=1)
            vertical_edges = np.sum(strong_edges, axis=0)
            
            # If we have strong horizontal or vertical lines, likely rectangular
            if np.max(horizontal_edges) > len(horizontal_edges) * 0.3 or np.max(vertical_edges) > len(vertical_edges) * 0.3:
                shapes_found.append("rectangular elements")
            
            # Check for circular patterns by looking at edge distribution
            height, width = img_array.shape
            center_y, center_x = height // 2, width // 2
            
            # Sample points in a circular pattern and check for edges
            circular_edge_count = 0
            total_samples = 0
            
            for angle in range(0, 360, 30):  # Sample every 30 degrees
                for radius in [min(height, width) // 6, min(height, width) // 4]:
                    x = int(center_x + radius * np.cos(np.radians(angle)))
                    y = int(center_y + radius * np.sin(np.radians(angle)))
                    
                    if 0 <= x < width and 0 <= y < height:
                        if strong_edges[y, x]:
                            circular_edge_count += 1
                        total_samples += 1
            
            if total_samples > 0 and circular_edge_count / total_samples > 0.3:
                shapes_found.append("circular elements")
                
        except Exception:
            # Fallback to basic classification
            shapes_found = ["geometric elements"]
        
        # Analyze aspect ratio for additional shape hints
        width, height = self.current_image.size
        aspect_ratio = width / height
        
        if abs(aspect_ratio - 1.0) < 0.1:
            shapes_found.append("square proportions")
        elif aspect_ratio > 2.0:
            shapes_found.append("elongated horizontal forms")
        elif aspect_ratio < 0.5:
            shapes_found.append("elongated vertical forms")
        
        return "Geometric shapes include " + ", ".join(shapes_found) if shapes_found else "Basic geometric forms present"
    
    def _analyze_organic_shapes(self) -> str:
        """Analyze organic shapes using curvature and complexity analysis"""
        if not self.current_image:
            return "Organic shapes with natural, flowing lines"
            
        # Convert to grayscale for organic shape analysis
        gray_image = self.current_image.convert('L')
        img_array = np.array(gray_image)
        
        organic_qualities = []
        
        try:
            # Calculate curvature using second derivatives
            gradient_x = np.gradient(img_array, axis=1)
            gradient_y = np.gradient(img_array, axis=0)
            
            # Second derivatives for curvature
            grad_xx = np.gradient(gradient_x, axis=1)
            grad_yy = np.gradient(gradient_y, axis=0)
            grad_xy = np.gradient(gradient_x, axis=0)
            
            # Curvature measure (simplified)
            curvature = np.sqrt(grad_xx**2 + grad_yy**2 + 2*grad_xy**2)
            avg_curvature = np.mean(curvature)
            curvature_variance = np.var(curvature)
            
            # Classify based on curvature characteristics
            if avg_curvature > 5:
                organic_qualities.append("highly curved")
            elif avg_curvature > 2:
                organic_qualities.append("moderately curved")
            else:
                organic_qualities.append("gently curved")
                
            if curvature_variance > 25:
                organic_qualities.append("irregular organic forms")
            elif curvature_variance > 10:
                organic_qualities.append("varied organic shapes")
            else:
                organic_qualities.append("smooth organic shapes")
                
        except Exception:
            # Fallback analysis
            organic_qualities.append("natural forms")
        
        # Analyze color distribution to infer natural vs artificial
        try:
            color_palette = self._extract_color_palette()
            natural_colors = ['green', 'brown', 'blue', 'gray', 'lightblue']
            color_names = [color[0] for color in color_palette.dominant_colors[:3]]
            
            natural_color_count = sum(1 for color in color_names if color in natural_colors)
            
            if natural_color_count >= 2:
                organic_qualities.append("natural coloring")
            else:
                organic_qualities.append("artificial coloring")
                
        except Exception:
            pass
        
        # Analyze edge smoothness
        try:
            # Calculate edge gradient smoothness
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            gradient_changes = np.abs(np.gradient(gradient_magnitude.flatten()))
            smoothness = 1.0 / (1.0 + np.mean(gradient_changes))
            
            if smoothness > 0.8:
                organic_qualities.append("flowing lines")
            elif smoothness > 0.5:
                organic_qualities.append("moderately flowing contours")
            else:
                organic_qualities.append("angular transitions")
                
        except Exception:
            organic_qualities.append("natural transitions")
        
        if not organic_qualities:
            return "Natural organic forms present"
        else:
            return "Organic shapes with " + ", ".join(organic_qualities)
    
    def _extract_color_palette(self) -> ColorPalette:
        """Extract dominant colors from the image with improved analysis"""
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
        
        # Convert to hex and name
        dominant_colors = []
        accent_colors = []
        hues = []
        
        for i, (count, rgb) in enumerate(sorted_colors[:10]):  # Get top 10 colors
            hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            color_name = self._get_color_name(rgb)
            
            # Calculate HSV for harmony analysis
            hsv = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
            hue = hsv[0] * 360  # Convert to degrees
            
            if i < 5:  # Top 5 are dominant
                dominant_colors.append((color_name, hex_color))
                hues.append(hue)
            else:  # Rest are accent colors
                accent_colors.append((color_name, hex_color))
        
        # Analyze color harmony
        color_harmony = self._analyze_color_harmony(hues)
        
        # Determine color temperature
        color_temperature = self._analyze_color_temperature(dominant_colors)
        
        return ColorPalette(
            dominant_colors=dominant_colors,
            accent_colors=accent_colors,
            color_harmony=color_harmony,
            color_temperature=color_temperature
        )
    
    def _analyze_color_harmony(self, hues: List[float]) -> str:
        """Analyze color harmony relationships"""
        if len(hues) < 2:
            return "monochromatic"
        
        # Calculate hue differences
        hue_diffs = []
        for i in range(len(hues)):
            for j in range(i+1, len(hues)):
                diff = abs(hues[i] - hues[j])
                # Handle circular nature of hue wheel
                if diff > 180:
                    diff = 360 - diff
                hue_diffs.append(diff)
        
        if not hue_diffs:
            return "monochromatic"
        
        avg_diff = np.mean(hue_diffs)
        max_diff = max(hue_diffs)
        
        # Classify harmony type
        if max_diff < 30:
            return "monochromatic"
        elif max_diff < 60:
            return "analogous"
        elif any(150 <= diff <= 210 for diff in hue_diffs):
            return "complementary"
        elif any(100 <= diff <= 140 for diff in hue_diffs):
            return "triadic"
        elif avg_diff > 90:
            return "tetradic"
        else:
            return "complex harmony"
    
    def _analyze_color_temperature(self, dominant_colors: List[Tuple[str, str]]) -> str:
        """Analyze overall color temperature"""
        warm_colors = ['red', 'orange', 'yellow', 'pink', 'brown']
        cool_colors = ['blue', 'green', 'cyan', 'purple', 'lightblue']
        neutral_colors = ['gray', 'white', 'black']
        
        warm_count = sum(1 for color_name, _ in dominant_colors if color_name in warm_colors)
        cool_count = sum(1 for color_name, _ in dominant_colors if color_name in cool_colors)
        neutral_count = sum(1 for color_name, _ in dominant_colors if color_name in neutral_colors)
        
        if warm_count > cool_count and warm_count > neutral_count:
            return "warm"
        elif cool_count > warm_count and cool_count > neutral_count:
            return "cool"
        else:
            return "neutral"
    
    def _get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Get a more accurate color name from RGB values"""
        r, g, b = rgb
        
        # More sophisticated color classification
        if r > 240 and g > 240 and b > 240:
            return "white"
        elif r < 20 and g < 20 and b < 20:
            return "black"
        elif r > 200 and g < 100 and b < 100:
            return "red"
        elif r < 100 and g > 200 and b < 100:
            return "green"
        elif r < 100 and g < 100 and b > 200:
            return "blue"
        elif r > 200 and g > 200 and b < 100:
            return "yellow"
        elif r > 150 and g < 100 and b > 150:
            return "magenta"
        elif r < 100 and g > 150 and b > 150:
            return "cyan"
        elif r > 150 and g > 100 and b < 100:
            return "orange"
        elif r > 100 and g < 150 and b > 100:
            return "purple"
        elif r > 100 and g > 150 and b < 100:
            return "olive"
        elif r > 200 and g > 150 and b > 150:
            return "pink"
        elif r > 100 and g > 100 and b < 100:
            return "brown"
        elif 100 <= r <= 180 and 100 <= g <= 180 and 100 <= b <= 180:
            return "gray"
        elif r < 150 and g > 180 and b > 200:
            return "lightblue"
        elif r > 180 and g > 200 and b < 150:
            return "lightgreen"
        else:
            # Determine based on dominant component
            max_val = max(r, g, b)
            if max_val == r:
                return "reddish"
            elif max_val == g:
                return "greenish"
            else:
                return "bluish"
    
    def _analyze_lighting(self) -> LightingAnalysis:
        """Analyze lighting conditions using brightness distribution"""
        if not self.current_image:
            return LightingAnalysis(
                light_source="natural light",
                light_direction="from above", 
                light_quality="soft",
                shadow_description="gentle shadows",
                highlight_description="subtle highlights"
            )
            
        # Convert to grayscale for brightness analysis
        gray_image = self.current_image.convert('L')
        img_array = np.array(gray_image)
        height, width = img_array.shape
        
        # Analyze brightness distribution across quadrants
        top_half = img_array[:height//2, :]
        bottom_half = img_array[height//2:, :]
        left_half = img_array[:, :width//2]
        right_half = img_array[:, width//2:]
        
        top_brightness = np.mean(top_half)
        bottom_brightness = np.mean(bottom_half)
        left_brightness = np.mean(left_half)
        right_brightness = np.mean(right_half)
        
        # Determine light direction
        if top_brightness > bottom_brightness + 20:
            light_direction = "from above"
        elif bottom_brightness > top_brightness + 20:
            light_direction = "from below"
        elif right_brightness > left_brightness + 20:
            light_direction = "from the right"
        elif left_brightness > right_brightness + 20:
            light_direction = "from the left"
        else:
            light_direction = "evenly distributed"
        
        # Analyze light quality based on contrast
        brightness_std = np.std(img_array)
        overall_brightness = np.mean(img_array)
        
        if brightness_std < 30:
            light_quality = "soft and diffused"
        elif brightness_std > 80:
            light_quality = "hard and dramatic"
        else:
            light_quality = "moderate contrast"
        
        # Determine light source type
        if overall_brightness > 180:
            light_source = "bright natural light"
        elif overall_brightness > 120:
            light_source = "natural daylight"
        elif overall_brightness > 80:
            light_source = "ambient light"
        else:
            light_source = "low light conditions"
        
        # Analyze shadows and highlights
        highlight_pixels = np.sum(img_array > 220)
        shadow_pixels = np.sum(img_array < 50)
        total_pixels = img_array.size
        
        if highlight_pixels / total_pixels > 0.1:
            highlight_description = "prominent bright highlights"
        elif highlight_pixels / total_pixels > 0.03:
            highlight_description = "subtle highlights"
        else:
            highlight_description = "minimal highlights"
            
        if shadow_pixels / total_pixels > 0.1:
            shadow_description = "deep shadows"
        elif shadow_pixels / total_pixels > 0.03:
            shadow_description = "gentle shadows"
        else:
            shadow_description = "minimal shadows"
        
        return LightingAnalysis(
            light_source=light_source,
            light_direction=light_direction,
            light_quality=light_quality,
            shadow_description=shadow_description,
            highlight_description=highlight_description
        )
    
    def _analyze_textures(self) -> str:
        """Analyze surface textures using image processing techniques"""
        if not self.current_image:
            return "smooth and varied surface textures"
            
        # Convert to grayscale for texture analysis
        gray_image = self.current_image.convert('L')
        img_array = np.array(gray_image)
        
        # Calculate local variance as a texture measure
        # High variance indicates rough/textured areas, low variance indicates smooth areas
        
        # Use a sliding window to calculate local variance
        height, width = img_array.shape
        window_size = min(20, height//10, width//10)  # Adaptive window size
        
        if window_size < 3:
            return "uniform surface texture"
        
        variances = []
        for y in range(0, height - window_size, window_size):
            for x in range(0, width - window_size, window_size):
                window = img_array[y:y+window_size, x:x+window_size]
                var = np.var(window)
                variances.append(var)
        
        if not variances:
            return "uniform surface texture"
            
        avg_variance = np.mean(variances)
        max_variance = max(variances)
        variance_std = np.std(variances)
        
        # Classify texture based on variance measures
        textures = []
        
        if avg_variance < 100:
            textures.append("smooth")
        elif avg_variance < 500:
            textures.append("moderately textured")
        else:
            textures.append("rough")
            
        if max_variance > 2000:
            textures.append("highly detailed areas")
        
        if variance_std > 300:
            textures.append("varied surface qualities")
        else:
            textures.append("consistent texture")
            
        # Analyze edge density for additional texture description
        try:
            # Simple edge detection using gradient
            gradient_x = np.gradient(img_array, axis=1)
            gradient_y = np.gradient(img_array, axis=0)
            edge_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            edge_density = np.mean(edge_magnitude)
            
            if edge_density > 15:
                textures.append("intricate surface details")
            elif edge_density > 8:
                textures.append("moderate surface detail")
            else:
                textures.append("minimal surface detail")
                
        except:
            # Fallback if gradient calculation fails
            pass
        
        return ", ".join(textures)
    
    def _identify_patterns(self) -> str:
        """Identify repeating patterns using frequency analysis"""
        if not self.current_image:
            return "no obvious repeating patterns"
            
        # Convert to grayscale for pattern analysis
        gray_image = self.current_image.convert('L')
        img_array = np.array(gray_image)
        height, width = img_array.shape
        
        patterns_found = []
        
        # Check for horizontal patterns (like stripes)
        horizontal_diffs = []
        for y in range(height):
            row = img_array[y, :]
            # Calculate differences between adjacent pixels
            diffs = np.abs(np.diff(row))
            horizontal_diffs.extend(diffs)
        
        horizontal_pattern_strength = np.var(horizontal_diffs)
        
        # Check for vertical patterns
        vertical_diffs = []
        for x in range(width):
            col = img_array[:, x]
            diffs = np.abs(np.diff(col))
            vertical_diffs.extend(diffs)
        
        vertical_pattern_strength = np.var(vertical_diffs)
        
        # Analyze pattern strength
        if horizontal_pattern_strength > 1000:
            patterns_found.append("horizontal striping")
        if vertical_pattern_strength > 1000:
            patterns_found.append("vertical striping")
        
        # Check for regular texture patterns using autocorrelation-like measure
        # Sample small regions and look for repetition
        sample_size = min(32, height//4, width//4)
        if sample_size > 8:
            try:
                # Take samples from different parts of the image
                top_left = img_array[:sample_size, :sample_size]
                top_right = img_array[:sample_size, -sample_size:]
                bottom_left = img_array[-sample_size:, :sample_size]
                bottom_right = img_array[-sample_size:, -sample_size:]
                
                samples = [top_left, top_right, bottom_left, bottom_right]
                correlations = []
                
                for i in range(len(samples)):
                    for j in range(i+1, len(samples)):
                        if samples[i].shape == samples[j].shape:
                            corr = np.corrcoef(samples[i].flatten(), samples[j].flatten())[0, 1]
                            if not np.isnan(corr):
                                correlations.append(abs(corr))
                
                if correlations and np.mean(correlations) > 0.7:
                    patterns_found.append("repeating texture elements")
                elif correlations and np.mean(correlations) > 0.4:
                    patterns_found.append("subtle pattern repetition")
                    
            except:
                # Skip pattern analysis if it fails
                pass
        
        # Check for grid-like patterns by analyzing regular spacing
        try:
            # Look for regular intensity changes that might indicate grids
            row_means = np.mean(img_array, axis=1)
            col_means = np.mean(img_array, axis=0)
            
            # Simple frequency analysis
            row_fft = np.abs(np.fft.fft(row_means))
            col_fft = np.abs(np.fft.fft(col_means))
            
            # Check if there are strong frequency components (indicating regularity)
            if len(row_fft) > 10 and np.max(row_fft[2:len(row_fft)//2]) > 3 * np.mean(row_fft[2:len(row_fft)//2]):
                patterns_found.append("regular horizontal spacing")
            if len(col_fft) > 10 and np.max(col_fft[2:len(col_fft)//2]) > 3 * np.mean(col_fft[2:len(col_fft)//2]):
                patterns_found.append("regular vertical spacing")
                
        except:
            # Skip if FFT analysis fails
            pass
        
        if not patterns_found:
            return "no obvious repeating patterns"
        else:
            return ", ".join(patterns_found)
    
    def _generate_level1_prompt(self) -> str:
        """Generate Level 1 core concept prompt following specification formula: [Style] of [Primary Subject] in a [Setting]"""
        if not self.analysis_results:
            # Analyze if not already done
            step1 = self._step1_initial_ingestion()
        else:
            step1 = self.analysis_results.get('step1')
        
        # Extract category and mood from step1 analysis
        category = self._classify_image_category()
        mood = self._assess_initial_mood()
        
        # Determine style based on image characteristics
        style = "A photograph"  # Default style
        
        # Determine primary subject based on category
        if category == "Portrait":
            primary_subject = "a person"
        elif category == "Landscape":
            primary_subject = "a natural landscape"
        elif category == "Architectural":
            primary_subject = "architectural elements"
        elif category == "Abstract":
            primary_subject = "abstract forms"
        elif category == "Still Life":
            primary_subject = "arranged objects"
        else:
            primary_subject = "various elements"
        
        # Determine setting based on mood and category
        if mood.lower() in ["peaceful", "serene"]:
            setting = "a tranquil setting"
        elif mood.lower() in ["dramatic", "chaotic"]:
            setting = "a dynamic environment"
        elif mood.lower() in ["melancholy", "mysterious"]:
            setting = "a subdued atmosphere"
        elif mood.lower() in ["joyful"]:
            setting = "a bright environment"
        else:
            setting = "a composed scene"
        
        return f"{style} of {primary_subject} in {setting}"


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