# Visual Analysis Agent

A comprehensive visual analysis system that implements the **Visual Deconstruction & Prompt Synthesis** agent as specified in `agent.md`.

## Overview

This implementation transforms images into detailed, structured textual descriptions through a systematic 5-step workflow and generates progressive prompts suitable for advanced text-to-image generation models.

## Features

### 5-Step Analysis Workflow

1. **Initial Ingestion & High-Level Triage**
   - Extract image metadata (dimensions, aspect ratio, file type)
   - Classify into high-level categories (Portrait, Landscape, etc.)
   - Assess initial mood and atmosphere

2. **Compositional & Structural Analysis**
   - Determine layout rules (Rule of Thirds, Golden Ratio, etc.)
   - Identify focal points and their positions
   - Analyze depth layers (foreground, midground, background)
   - Evaluate negative space usage

3. **Shape & Form Deconstruction**
   - Break down objects into geometric shapes
   - Describe organic forms with evocative language
   - Create a comprehensive "shape inventory"

4. **Color Palette & Lighting Analysis**
   - Extract dominant and accent colors with hex codes
   - Analyze color harmony relationships
   - Identify lighting sources, direction, and quality
   - Describe shadows and highlights

5. **Texture & Material Definition**
   - Analyze surface textures with tactile adjectives
   - Identify repeating patterns
   - Create material specification sheets

### Progressive Prompt Generation

The system generates 4 levels of increasingly detailed prompts:

- **Level 1: Core Concept** - Essential image description
- **Level 2: Detailed Composition** - Adds structural information
- **Level 3: Artistic & Atmospheric** - Includes mood, lighting, and colors  
- **Level 4: Master Blueprint** - Comprehensive technical specifications

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The system requires:
   - Python 3.7+
   - Pillow (PIL) for image processing
   - NumPy for numerical operations

## Usage

### Basic Usage

```python
from visual_analysis_agent import VisualAnalysisAgent

# Initialize the agent
agent = VisualAnalysisAgent()

# Analyze an image
results = agent.analyze_image('path/to/your/image.jpg')

# Access analysis results
print(results['step1'].output)  # Initial triage
print(results['step2'].output)  # Composition analysis
# ... etc for steps 3-5

# Get progressive prompts
prompts = results['prompts']
print(prompts['level1'])  # Core concept
print(prompts['level4'])  # Master blueprint
```

### Command Line Interface

Use the CLI for easy analysis:

```bash
# Full analysis
python cli.py path/to/your/image.jpg

# Show only specific prompt level
python cli.py image.jpg --level 4

# Show only specific analysis step
python cli.py image.jpg --step 1 --verbose

# JSON output for programmatic use
python cli.py image.jpg --json
```

### Demo Script

Run the included demo script:

```bash
# With your own image
python demo.py path/to/your/image.jpg

# With auto-generated sample image
python demo.py
```

The demo will:
- Create a sample landscape image (if no image provided)
- Run the complete 5-step analysis
- Display structured results
- Show all 4 prompt levels

## Implementation Details

### Core Classes

- **`VisualAnalysisAgent`**: Main agent class implementing the 5-step workflow
- **`ImageMetadata`**: Container for basic image properties
- **`ColorPalette`**: Color analysis results with dominant/accent colors
- **`LightingAnalysis`**: Lighting conditions and quality assessment
- **`AnalysisStep`**: Base structure for analysis step results

### Analysis Algorithms

The implementation provides sophisticated computer vision algorithms:
- **Advanced color analysis** with harmony detection (analogous, complementary, triadic) and temperature classification
- **Composition rule detection** using variance analysis for Rule of Thirds, Golden Ratio, symmetry detection
- **Intelligent focal point mapping** with 9-region positioning analysis
- **Depth layer analysis** using brightness and sharpness distribution
- **Texture analysis** with local variance and edge density measurements
- **Pattern detection** using FFT analysis and autocorrelation techniques  
- **Shape deconstruction** with geometric and organic form analysis using curvature detection
- **Lighting analysis** with quadrant-based direction detection and quality assessment
- **Mood classification** based on brightness variance and color psychology
- **Category detection** using aspect ratio, color distribution, and compositional heuristics

### Extensibility

The system is designed for easy enhancement with:
- Computer vision libraries (OpenCV, scikit-image)
- Machine learning models for classification
- Advanced color analysis algorithms
- Sophisticated composition detection
- Deep learning-based feature extraction

## Example Output

```
STEP 1: INITIAL INGESTION & HIGH-LEVEL TRIAGE
0.75:1 aspect ratio portrait. Joyful mood.

STEP 2: COMPOSITIONAL & STRUCTURAL ANALYSIS  
Composition follows rule of thirds. Focal point positioned in the lower-right area
Strong depth separation with detailed background, sharp midground, shadowed foreground

STEP 4: COLOR PALETTE & LIGHTING ANALYSIS  
Color palette dominated by white, pink, bluish. 
Lighting: hard and dramatic bright natural light from above.

STEP 5: TEXTURE & MATERIAL DEFINITION
Textures: rough, highly detailed areas, varied surface qualities.
Patterns: horizontal striping, vertical striping, subtle pattern repetition

LEVEL 1 PROMPT: A photograph of a person in a bright environment

LEVEL 4 PROMPT: A photograph of a person in a bright environment, rule of thirds, 
focal point positioned in the lower-right area strong depth separation with detailed 
background, sharp midground, shadowed foreground. Lighting: hard and dramatic bright 
natural light from above. Color palette dominated by white, pink, bluish, creating 
a joyful mood. rough, highly detailed areas, varied surface qualities, patterns: 
horizontal striping, vertical striping, subtle pattern repetition. Geometric shapes 
include simple geometric forms, rectangular elements Organic shapes with moderately 
curved, irregular organic forms, artificial coloring, angular transitions. Shot with 
sharp focus, high dynamic range, professional lighting
```

## Architecture

The implementation follows the exact specification from `agent.md`:

```
Image Input → 5-Step Analysis → Progressive Prompts
     ↓              ↓                    ↓
 Metadata    Structured Data    4 Prompt Levels
```

Each step builds upon the previous one to create a comprehensive visual profile suitable for AI image generation systems.

## Future Enhancements

The implementation is now production-ready but could be enhanced further with:

- **Deep Learning Integration**: Add trained models for object detection (YOLO/R-CNN)
- **Advanced Computer Vision**: Integrate OpenCV for contour detection and advanced filtering
- **Semantic Understanding**: Include scene understanding and object relationship analysis  
- **Style Transfer Analysis**: Add artistic style classification capabilities
- **Performance Optimization**: Implement caching and multi-threading for large images
- **Extended Format Support**: Add support for RAW, TIFF, and other professional formats
- **Batch Processing**: Add capabilities for analyzing multiple images simultaneously
- **Interactive Features**: Web interface for real-time image analysis
- **Custom Training**: Allow fine-tuning of classification models for specific domains

## License

This implementation is based on the agent specification provided in `agent.md`.