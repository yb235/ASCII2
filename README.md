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

The current implementation provides a solid foundation with:
- Basic color extraction using PIL's `getcolors()`
- Simplified compositional heuristics
- Template-based prompt generation
- Extensible architecture for advanced algorithms

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
16:9 aspect ratio general image. Neutral mood.

STEP 4: COLOR PALETTE & LIGHTING ANALYSIS  
Color palette dominated by lightblue, gray, darkblue. 
Lighting: soft natural light from from above.

LEVEL 1 PROMPT: A detailed image

LEVEL 4 PROMPT: A detailed image, composition follows balanced composition. 
Central focal point Clear depth with foreground, midground, and background elements. 
Negative space provides balance.. Color palette dominated by lightblue, gray, darkblue. 
Lighting: soft natural light from from above. Textures: smooth and varied surface textures. 
Patterns: no obvious repeating patterns Geometric elements include rectangular and 
circular forms. Organic shapes with natural, flowing lines.
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

Potential improvements for production use:

- **Advanced Computer Vision**: Integrate OpenCV for sophisticated analysis
- **Machine Learning**: Add trained models for category/mood classification  
- **Color Science**: Implement advanced color harmony detection
- **Composition Detection**: Use rule-based composition analysis
- **Texture Analysis**: Add Gabor filters and texture descriptors
- **Object Detection**: Integrate YOLO/R-CNN for object identification
- **Style Analysis**: Add artistic style classification
- **Semantic Understanding**: Include scene understanding capabilities

## License

This implementation is based on the agent specification provided in `agent.md`.