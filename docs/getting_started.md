# Getting Started

This guide will help you get started with the CAPA SDK.

## Installation

### From PyPI (Recommended)

```bash
pip install capa-sdk
```

### From Source

```bash
git clone https://github.com/yarel2l/capa-sdk.git
cd capa-sdk
pip install -e .
```

### Dependencies

The SDK will automatically install required dependencies:

- numpy >= 1.24.0
- scipy >= 1.11.0
- opencv-python >= 4.8.0
- Pillow >= 10.0.0
- dlib >= 19.24.0
- face-recognition >= 1.3.0
- mediapipe >= 0.10.0
- scikit-learn >= 1.3.0

**Note:** dlib may require additional system dependencies. On macOS:

```bash
brew install cmake
```

On Ubuntu/Debian:

```bash
sudo apt-get install cmake libboost-all-dev
```

### Required Model File

The SDK requires the dlib 68-point facial landmark model. Download and extract it:

```bash
# Download the model
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# Extract it
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# Move to your project directory or a location in your PATH
mv shape_predictor_68_face_landmarks.dat /path/to/your/project/
```

Alternatively, you can download it manually from:
- [dlib model files](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

**Note:** This file is ~68 MB and is required for facial landmark detection.

## Quick Start

### Basic Analysis

```python
from capa import CoreAnalyzer

# Initialize the analyzer
analyzer = CoreAnalyzer()

# Analyze an image
result = analyzer.analyze_image("path/to/photo.jpg")

# Check if analysis was successful
if result is not None:
    # Access WD analysis
    if result.wd_result:
        print(f"WD Value: {result.wd_result.wd_value:.3f}")
        print(f"Classification: {result.wd_result.primary_classification.value}")
        print(f"Confidence: {result.wd_result.measurement_confidence*100:.1f}%")

    # Access Forehead analysis
    if result.forehead_result:
        print(f"Slant Angle: {result.forehead_result.forehead_geometry.slant_angle_degrees:.1f}°")
        print(f"Impulsiveness: {result.forehead_result.impulsiveness_level.value}")

    # Access Morphology analysis
    if result.morphology_result:
        print(f"Face Shape: {result.morphology_result.shape_classification.primary_shape.value}")
        print(f"Facial Index: {result.morphology_result.facial_proportions.facial_index:.1f}")

# Always shutdown to release resources
analyzer.shutdown()
```

### Using Analysis Modes

```python
from capa import CoreAnalyzer, AnalysisConfiguration, AnalysisMode

# Configure for fast analysis
config = AnalysisConfiguration(
    mode=AnalysisMode.FAST,
    enable_wd_analysis=True,
    enable_forehead_analysis=True,
    enable_morphology_analysis=True,
)

analyzer = CoreAnalyzer(config=config)
result = analyzer.analyze_image("photo.jpg")
analyzer.shutdown()
```

### Available Analysis Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `FAST` | Quick analysis with basic processing | Real-time applications |
| `STANDARD` | Balanced analysis (default) | General use |
| `THOROUGH` | Deep analysis with all features | Detailed reports |
| `SCIENTIFIC` | Maximum accuracy, 2D observables only | Research |
| `RESEARCH` | Includes peer-reviewed correlations | Academic studies |
| `REALTIME` | Optimized for video streams | Live processing |

### Analyzing Multiple Images

```python
from capa import MultiAngleAnalyzer, AngleSpecification

analyzer = MultiAngleAnalyzer()

# Define angle specifications
specs = [
    AngleSpecification(angle_type='frontal', image_path='front.jpg'),
    AngleSpecification(angle_type='lateral_left', image_path='left.jpg'),
    AngleSpecification(angle_type='profile', image_path='profile.jpg'),
]

# Analyze
result = analyzer.analyze_multiple_angles(
    angle_specs=specs,
    subject_id="subject_001"
)

# Combined results
print(f"Combined Confidence: {result.combined_confidence*100:.1f}%")
print(f"Combined WD: {result.combined_wd_value}")

analyzer.shutdown()
```

## Using Individual Modules

You can use individual analysis modules directly for more control:

```python
import cv2
from capa.modules import WDAnalyzer, ForeheadAnalyzer, MorphologyAnalyzer

# Load image
image = cv2.imread("photo.jpg")

# WD Analysis
wd_analyzer = WDAnalyzer()
wd_result = wd_analyzer.analyze_image(image)
if wd_result:
    print(f"WD: {wd_result.wd_value:.3f}")

# Forehead Analysis
forehead_analyzer = ForeheadAnalyzer()
forehead_result = forehead_analyzer.analyze_image(image)
if forehead_result:
    print(f"Angle: {forehead_result.forehead_geometry.slant_angle_degrees:.1f}°")

# Morphology Analysis
morphology_analyzer = MorphologyAnalyzer()
morphology_result = morphology_analyzer.analyze_image(image)
if morphology_result:
    print(f"Shape: {morphology_result.shape_classification.primary_shape.value}")
```

## Error Handling

```python
from capa import CoreAnalyzer

analyzer = CoreAnalyzer()

try:
    result = analyzer.analyze_image("photo.jpg")

    if result is None:
        print("No face detected or analysis failed")
    elif result.processing_metadata.overall_confidence < 0.5:
        print("Low confidence result - image quality may be poor")
    else:
        # Process results
        pass

except FileNotFoundError:
    print("Image file not found")
except Exception as e:
    print(f"Analysis error: {e}")
finally:
    analyzer.shutdown()
```

## Next Steps

- Read the [API Reference](api_reference.md) for detailed documentation
- Explore [Examples](examples.md) for more use cases
- Learn about the [Scientific Foundation](scientific_foundation.md)
- Configure advanced options in [Configuration](configuration.md)
