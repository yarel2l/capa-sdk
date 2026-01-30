# Examples

This section provides practical examples for common CAPA SDK use cases.

## Basic Analysis

### Single Image Analysis

```python
from capa import CoreAnalyzer

# Initialize analyzer
analyzer = CoreAnalyzer()

# Analyze an image
result = analyzer.analyze_image("photo.jpg")

if result:
    print(f"Overall confidence: {result.processing_metadata.overall_confidence*100:.1f}%")

    # WD Analysis
    if result.wd_result:
        print(f"WD Value: {result.wd_result.wd_value:.3f}")
        print(f"Classification: {result.wd_result.primary_classification.value}")

    # Forehead Analysis
    if result.forehead_result:
        print(f"Slant Angle: {result.forehead_result.forehead_geometry.slant_angle_degrees:.1f}°")
        print(f"Impulsiveness: {result.forehead_result.impulsiveness_level.value}")

    # Morphology Analysis
    if result.morphology_result:
        print(f"Face Shape: {result.morphology_result.shape_classification.primary_shape.value}")

# Clean up
analyzer.shutdown()
```

### Analysis with Custom Configuration

```python
from capa import CoreAnalyzer, AnalysisConfiguration, AnalysisMode

# Configure for thorough analysis
config = AnalysisConfiguration(
    mode=AnalysisMode.THOROUGH,
    enable_wd_analysis=True,
    enable_forehead_analysis=True,
    enable_morphology_analysis=True,
)

analyzer = CoreAnalyzer(config=config)
result = analyzer.analyze_image("photo.jpg")
analyzer.shutdown()
```

---

## Multi-Angle Analysis

### Basic Multi-Angle

```python
from capa import MultiAngleAnalyzer, AngleSpecification

analyzer = MultiAngleAnalyzer()

# Define images with their angles
specs = [
    AngleSpecification(angle_type='frontal', image_path='front.jpg'),
    AngleSpecification(angle_type='lateral_left', image_path='left.jpg'),
    AngleSpecification(angle_type='lateral_right', image_path='right.jpg'),
]

result = analyzer.analyze_multiple_angles(
    angle_specs=specs,
    subject_id="subject_001"
)

print(f"Combined Confidence: {result.combined_confidence*100:.1f}%")
print(f"Angles Analyzed: {list(result.angle_results.keys())}")

analyzer.shutdown()
```

### Available Angle Types

| Angle Type | Description | Best For |
|------------|-------------|----------|
| `frontal` | Direct front view | WD, Morphology |
| `lateral_left` | Left side view | Forehead analysis |
| `lateral_right` | Right side view | Forehead analysis |
| `profile` | Full side profile | Forehead geometry |
| `three_quarter_left` | 45° left | Combined analysis |
| `three_quarter_right` | 45° right | Combined analysis |

---

## Individual Modules

### WD Analyzer

```python
import cv2
from capa.modules import WDAnalyzer

image = cv2.imread("photo.jpg")
analyzer = WDAnalyzer()

result = analyzer.analyze_image(image)

if result:
    print(f"WD Value: {result.wd_value:.3f}")
    print(f"Bizygomatic Width: {result.bizygomatic_width:.1f}px")
    print(f"Bigonial Width: {result.bigonial_width:.1f}px")
    print(f"Classification: {result.primary_classification.value}")

    # Classification details
    for cls in result.classifications:
        print(f"  {cls.classification.value}: {cls.confidence*100:.1f}%")
```

### Forehead Analyzer

```python
import cv2
from capa.modules import ForeheadAnalyzer

image = cv2.imread("profile.jpg")
analyzer = ForeheadAnalyzer()

result = analyzer.analyze_image(image)

if result:
    geometry = result.forehead_geometry
    print(f"Slant Angle: {geometry.slant_angle_degrees:.1f}°")
    print(f"Forehead Height: {geometry.forehead_height:.1f}px")
    print(f"Forehead Width: {geometry.forehead_width:.1f}px")
    print(f"Forehead Curvature: {geometry.forehead_curvature:.2f}")
    print(f"Impulsiveness Level: {result.impulsiveness_level.value}")
```

### Morphology Analyzer

```python
import cv2
from capa.modules import MorphologyAnalyzer

image = cv2.imread("photo.jpg")
analyzer = MorphologyAnalyzer()

result = analyzer.analyze_image(image)

if result:
    # Face shape
    shape = result.shape_classification
    print(f"Primary Shape: {shape.primary_shape.value}")
    print(f"Shape Confidence: {shape.confidence*100:.1f}%")

    # Facial proportions
    props = result.facial_proportions
    print(f"Facial Index: {props.facial_index:.1f}")
    print(f"Width/Height Ratio: {props.facial_width_height_ratio:.3f}")
```

### Neoclassical Canons Analyzer

```python
import cv2
from capa.modules import NeoclassicalCanonsAnalyzer

image = cv2.imread("photo.jpg")
analyzer = NeoclassicalCanonsAnalyzer()

result = analyzer.analyze_image(image)

if result:
    print(f"Overall Harmony: {result.overall_harmony_score*100:.1f}%")

    # Individual canon compliance
    for canon in result.canon_results:
        print(f"Canon {canon.canon_number}: {canon.name}")
        print(f"  Compliance: {canon.compliance_percentage:.1f}%")
        print(f"  Deviation: {canon.deviation:.3f}")
```

---

## Batch Processing

### Processing Multiple Images

```python
from pathlib import Path
from capa import CoreAnalyzer

analyzer = CoreAnalyzer()
image_dir = Path("images/")

results = {}
for image_path in image_dir.glob("*.jpg"):
    result = analyzer.analyze_image(str(image_path))
    if result:
        results[image_path.name] = {
            'confidence': result.processing_metadata.overall_confidence,
            'wd_value': result.wd_result.wd_value if result.wd_result else None,
            'face_shape': result.morphology_result.shape_classification.primary_shape.value
                         if result.morphology_result else None,
        }

analyzer.shutdown()

# Print summary
for name, data in results.items():
    print(f"{name}: WD={data['wd_value']:.3f}, Shape={data['face_shape']}")
```

---

## Working with NumPy Arrays

### Direct Array Analysis

```python
import cv2
import numpy as np
from capa import CoreAnalyzer

# Load image as numpy array
image = cv2.imread("photo.jpg")

# Or create from other sources
# image = np.array(pil_image)

analyzer = CoreAnalyzer()

# analyze_image accepts both file paths and numpy arrays
result = analyzer.analyze_image(image)

analyzer.shutdown()
```

---

## Error Handling

### Robust Analysis Pipeline

```python
from capa import CoreAnalyzer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_safely(image_path: str) -> dict:
    """Analyze an image with comprehensive error handling."""
    analyzer = None

    try:
        analyzer = CoreAnalyzer()
        result = analyzer.analyze_image(image_path)

        if result is None:
            logger.warning(f"No face detected in {image_path}")
            return {'success': False, 'error': 'No face detected'}

        if result.processing_metadata.overall_confidence < 0.5:
            logger.warning(f"Low confidence result for {image_path}")
            return {
                'success': True,
                'warning': 'Low confidence',
                'data': result
            }

        return {'success': True, 'data': result}

    except FileNotFoundError:
        logger.error(f"Image not found: {image_path}")
        return {'success': False, 'error': 'File not found'}

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {'success': False, 'error': str(e)}

    finally:
        if analyzer:
            analyzer.shutdown()
```

---

## Integration Examples

### Flask Web Service

```python
from flask import Flask, request, jsonify
from capa import CoreAnalyzer
import tempfile
import os

app = Flask(__name__)
analyzer = CoreAnalyzer()

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        file.save(tmp.name)

        try:
            result = analyzer.analyze_image(tmp.name)

            if result is None:
                return jsonify({'error': 'No face detected'}), 400

            return jsonify({
                'confidence': result.processing_metadata.overall_confidence,
                'wd_value': result.wd_result.wd_value if result.wd_result else None,
                'face_shape': result.morphology_result.shape_classification.primary_shape.value
                             if result.morphology_result else None,
            })

        finally:
            os.unlink(tmp.name)

@app.teardown_appcontext
def shutdown_analyzer(exception=None):
    analyzer.shutdown()

if __name__ == '__main__':
    app.run(debug=True)
```

### Command Line Tool

```python
#!/usr/bin/env python
"""Simple CLI for CAPA analysis."""

import argparse
import json
from capa import CoreAnalyzer, AnalysisMode, AnalysisConfiguration

def main():
    parser = argparse.ArgumentParser(description='CAPA Facial Analysis')
    parser.add_argument('image', help='Path to image file')
    parser.add_argument('--mode', choices=['fast', 'standard', 'thorough'],
                       default='standard', help='Analysis mode')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    modes = {
        'fast': AnalysisMode.FAST,
        'standard': AnalysisMode.STANDARD,
        'thorough': AnalysisMode.THOROUGH,
    }

    config = AnalysisConfiguration(mode=modes[args.mode])
    analyzer = CoreAnalyzer(config=config)

    try:
        result = analyzer.analyze_image(args.image)

        if result is None:
            print("No face detected")
            return 1

        if args.json:
            output = {
                'confidence': result.processing_metadata.overall_confidence,
                'wd': {
                    'value': result.wd_result.wd_value,
                    'classification': result.wd_result.primary_classification.value,
                } if result.wd_result else None,
                'forehead': {
                    'angle': result.forehead_result.forehead_geometry.slant_angle_degrees,
                    'impulsiveness': result.forehead_result.impulsiveness_level.value,
                } if result.forehead_result else None,
                'morphology': {
                    'shape': result.morphology_result.shape_classification.primary_shape.value,
                    'facial_index': result.morphology_result.facial_proportions.facial_index,
                } if result.morphology_result else None,
            }
            print(json.dumps(output, indent=2))
        else:
            print(f"Confidence: {result.processing_metadata.overall_confidence*100:.1f}%")
            if result.wd_result:
                print(f"WD: {result.wd_result.wd_value:.3f} ({result.wd_result.primary_classification.value})")
            if result.forehead_result:
                print(f"Forehead: {result.forehead_result.forehead_geometry.slant_angle_degrees:.1f}°")
            if result.morphology_result:
                print(f"Shape: {result.morphology_result.shape_classification.primary_shape.value}")

        return 0

    finally:
        analyzer.shutdown()

if __name__ == '__main__':
    exit(main())
```

---

## Next Steps

- See [Configuration](configuration.md) for advanced options
- Read the [API Reference](api_reference.md) for complete documentation
- Review [Scientific Foundation](scientific_foundation.md) for methodology details
