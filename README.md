# ASCII2: Advanced Visual Analysis & ASCII Art Generation

A comprehensive visual analysis system that combines sophisticated image analysis with high-quality ASCII art conversion. The project implements the **Visual Deconstruction & Prompt Synthesis** agent with enhanced ASCII art capabilities for creating detailed textual representations of images.

## ✨ Key Features

### 🎨 High-Quality ASCII Art Conversion
- **Multiple Character Sets**: Simple (10 chars), Extended (64 chars), Blocks (4 chars), Detailed (64 chars)
- **Quality Presets**: Low, Medium, High, and Ultra quality levels
- **Advanced Preprocessing**: Contrast enhancement, edge detection, and sharpening
- **Intelligent Scaling**: Aspect ratio preservation with character dimension adjustment
- **Performance Optimized**: Efficient numpy-based processing

### 🔍 5-Step Visual Analysis Workflow
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
   - Create comprehensive "shape inventory"

4. **Color Palette & Lighting Analysis**
   - Extract dominant and accent colors with hex codes
   - Analyze color harmony relationships
   - Identify lighting sources, direction, and quality
   - Describe shadows and highlights

5. **Texture & Material Definition**
   - Analyze surface textures with tactile adjectives
   - Identify repeating patterns
   - Create material specification sheets

### 📝 Progressive Prompt Generation
The system generates 4 levels of increasingly detailed prompts:
- **Level 1: Core Concept** - Essential image description
- **Level 2: Detailed Composition** - Adds structural information
- **Level 3: Artistic & Atmospheric** - Includes mood, lighting, and colors  
- **Level 4: Master Blueprint** - Comprehensive technical specifications

## 📁 Project Structure

```
ASCII2/
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── .gitignore               # Git ignore rules
│
├── src/                     # Core application modules
│   ├── visual_analysis_agent.py      # Main visual analysis engine
│   ├── ascii_converter.py            # ASCII art conversion
│   ├── enhanced_visual_agent.py      # Enhanced analysis with ASCII
│   ├── enhanced_ascii_converter.py   # Advanced ASCII features
│   ├── ai_ascii.py                   # AI-enhanced ASCII processing
│   ├── simplified_hq_ascii.py        # Simplified high-quality converter
│   ├── ultimate_ascii.py             # Ultimate quality ASCII art
│   └── video_ascii.py                # Video ASCII conversion
│
├── cli/                     # Command line interfaces
│   ├── cli.py                        # Main CLI for analysis
│   └── demo.py                       # Interactive demonstration
│
├── tests/                   # Test files
│   ├── test_enhanced_features.py     # Comprehensive test suite
│   └── test_agent.py                 # Basic agent tests
│
├── docs/                    # Documentation
│   ├── agent.md                      # Detailed agent specification
│   ├── IMPLEMENTATION_SUMMARY.md     # Implementation details
│   ├── AI_ENHANCEMENT_SUMMARY.md     # AI enhancement details
│   └── COLOR_AI_SUCCESS.md           # Color processing achievements
│
└── examples/                # Sample data and outputs
    ├── WIN_20250919_19_52_29_Pro.jpg # Sample image
    ├── *.txt                         # ASCII art outputs
    └── *.html                        # HTML visualizations
```

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yb235/ASCII2.git
cd ASCII2
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **System requirements:**
   - Python 3.7+
   - Pillow (PIL) for image processing
   - NumPy for numerical operations

## 💻 Usage

### Command Line Interface

**Basic image analysis:**
```bash
python cli/cli.py examples/WIN_20250919_19_52_29_Pro.jpg
```

**Show specific prompt level:**
```bash
python cli/cli.py examples/image.jpg --level 4
```

**Show specific analysis step:**
```bash
python cli/cli.py examples/image.jpg --step 1 --verbose
```

**JSON output for programmatic use:**
```bash
python cli/cli.py examples/image.jpg --json
```

### ASCII Art Conversion

**High-quality ASCII conversion:**
```bash
python src/ascii_converter.py examples/image.jpg -o output.txt -q high
```

**Ultra quality with information:**
```bash
python src/ascii_converter.py examples/image.jpg -q ultra --info
```

**Custom settings:**
```bash
python src/ascii_converter.py examples/image.jpg -w 200 -c detailed --no-contrast
```

### Python API

**Basic visual analysis:**
```python
import sys
sys.path.append('src')
from visual_analysis_agent import VisualAnalysisAgent

# Initialize the agent
agent = VisualAnalysisAgent()

# Analyze an image
results = agent.analyze_image('examples/your_image.jpg')

# Access analysis results
print(results['step1'].output)  # Initial triage
print(results['step2'].output)  # Composition analysis

# Get progressive prompts
prompts = results['prompts']
print(prompts['level1'])  # Core concept
print(prompts['level4'])  # Master blueprint
```

**Enhanced analysis with ASCII:**
```python
from enhanced_visual_agent import EnhancedVisualAnalysisAgent

# Initialize enhanced agent
agent = EnhancedVisualAnalysisAgent(enable_ascii=True, ascii_quality='high')

# Run complete analysis
results = agent.analyze_image_enhanced('examples/your_image.jpg')

# Access ASCII art and analysis
print(results['ascii_metadata']['character_count'])
print(results['analysis']['prompts']['level4'])
```

### Demo Script

Run the interactive demonstration:
```bash
python cli/demo.py examples/WIN_20250919_19_52_29_Pro.jpg
```

## 📊 Example Output

```
STEP 1: INITIAL INGESTION & HIGH-LEVEL TRIAGE
1.78:1 aspect ratio landscape. Dynamic architectural mood.

STEP 2: COMPOSITIONAL & STRUCTURAL ANALYSIS  
Composition follows rule of thirds. Focal point positioned in the center area
Strong depth separation with detailed background, sharp midground

STEP 4: COLOR PALETTE & LIGHTING ANALYSIS  
Color palette dominated by blue, gray, white tones
Lighting: natural daylight with moderate contrast

STEP 5: TEXTURE & MATERIAL DEFINITION
Textures: smooth glass surfaces, rough concrete, metallic details
Patterns: geometric architectural elements, linear structural components

LEVEL 1 PROMPT: A photograph of architectural elements in a dynamic environment

LEVEL 4 PROMPT: A photograph of modern architectural elements in a dynamic 
urban environment, rule of thirds composition, focal point positioned centrally, 
strong depth separation with detailed background and sharp midground. Natural 
daylight with moderate contrast. Color palette dominated by blue, gray, and white 
tones creating a dynamic architectural mood. Smooth glass surfaces, rough concrete 
textures, metallic details, geometric patterns and linear structural components.
```

## 🏗️ Architecture

The system follows a modular architecture with clear separation of concerns:

```
Image Input → ASCII Conversion → 5-Step Analysis → Progressive Prompts
     ↓              ↓                    ↓                    ↓
 Metadata    ASCII Metadata     Structured Data        4 Prompt Levels
```

### Core Components

- **`VisualAnalysisAgent`**: Main analysis engine implementing the 5-step workflow
- **`ASCIIConverter`**: High-performance ASCII art generation
- **`EnhancedVisualAnalysisAgent`**: Integrated analysis with ASCII capabilities
- **`ImageMetadata`**: Container for image properties and analysis metadata
- **`ColorPalette`**: Advanced color analysis and harmony detection
- **`LightingAnalysis`**: Sophisticated lighting condition assessment

## 🎯 Recent Enhancements (2024-2025)

### ASCII Art Integration & Quality Improvements
- **High-Quality ASCII Art Converter**: Multiple character sets and quality presets
- **Advanced Preprocessing**: Contrast enhancement and edge detection
- **Intelligent Aspect Ratio Preservation**: Professional output formatting
- **Performance Optimizations**: Efficient numpy-based processing

### Enhanced Visual Analysis
- **Integrated Workflow**: Seamless ASCII art + visual analysis
- **Improved Error Handling**: Comprehensive validation and recovery
- **Batch Processing**: Multiple image analysis capabilities
- **Export Functionality**: JSON export and detailed reporting

### Code Quality Improvements
- **Better Error Handling**: Specific exception types and graceful recovery
- **Input Validation**: Image format verification and file existence checks
- **Performance Monitoring**: Processing time tracking and optimization
- **Enhanced Documentation**: Comprehensive docstrings and type hints

## 🧪 Testing

Run the comprehensive test suite:
```bash
python tests/test_enhanced_features.py
```

Run basic agent tests:
```bash
python tests/test_agent.py
```

## 🚀 Future Enhancements

- **Real-time ASCII Animation**: Video file processing for animated ASCII
- **Interactive Web Interface**: Live image analysis and ASCII generation
- **Machine Learning Integration**: Trained models for enhanced object detection
- **Advanced Computer Vision**: OpenCV integration for sophisticated analysis
- **Color ASCII Art**: ANSI escape sequences for colorized output
- **Custom Training**: Fine-tuning capabilities for specific domains

## 📄 License

This implementation is based on the agent specification provided in `docs/agent.md`. The project is open source and available for educational and research purposes.

## 🔗 Related Documentation

- [`docs/agent.md`](docs/agent.md) - Detailed agent specification and methodology
- [`docs/IMPLEMENTATION_SUMMARY.md`](docs/IMPLEMENTATION_SUMMARY.md) - Technical implementation details
- [`docs/AI_ENHANCEMENT_SUMMARY.md`](docs/AI_ENHANCEMENT_SUMMARY.md) - AI enhancement documentation
- [`docs/COLOR_AI_SUCCESS.md`](docs/COLOR_AI_SUCCESS.md) - Color processing achievements

---

*Built with Python, PIL, and NumPy for high-performance image analysis and ASCII art generation.*