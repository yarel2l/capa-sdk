# Configuration

This guide covers all configuration options available in the CAPA SDK.

## Analysis Configuration

The `AnalysisConfiguration` class controls how analysis is performed.

### Basic Configuration

```python
from capa import AnalysisConfiguration, AnalysisMode

config = AnalysisConfiguration(
    mode=AnalysisMode.STANDARD,
    enable_wd_analysis=True,
    enable_forehead_analysis=True,
    enable_morphology_analysis=True,
)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `AnalysisMode` | `STANDARD` | Analysis mode (speed vs accuracy) |
| `enable_wd_analysis` | `bool` | `True` | Enable WD (bizygomatic) analysis |
| `enable_forehead_analysis` | `bool` | `True` | Enable forehead/impulsiveness analysis |
| `enable_morphology_analysis` | `bool` | `True` | Enable face shape analysis |
| `min_confidence_threshold` | `float` | `0.5` | Minimum confidence to return results |
| `enable_quality_checks` | `bool` | `True` | Enable image quality validation |

---

## Analysis Modes

### Available Modes

```python
from capa import AnalysisMode

# Fast mode - optimized for speed
AnalysisMode.FAST

# Standard mode - balanced (default)
AnalysisMode.STANDARD

# Thorough mode - maximum detail
AnalysisMode.THOROUGH

# Scientific mode - 2D observables only
AnalysisMode.SCIENTIFIC

# Research mode - includes all correlations
AnalysisMode.RESEARCH

# Realtime mode - optimized for video
AnalysisMode.REALTIME
```

### Mode Comparison

| Mode | Speed | Accuracy | Detectors | Use Case |
|------|-------|----------|-----------|----------|
| `FAST` | ★★★★★ | ★★★☆☆ | 1-2 | Real-time apps |
| `STANDARD` | ★★★★☆ | ★★★★☆ | 2-3 | General use |
| `THOROUGH` | ★★★☆☆ | ★★★★★ | All | Detailed reports |
| `SCIENTIFIC` | ★★★☆☆ | ★★★★★ | All | Research (2D only) |
| `RESEARCH` | ★★☆☆☆ | ★★★★★ | All | Academic studies |
| `REALTIME` | ★★★★★ | ★★☆☆☆ | 1 | Video streams |

### Mode Examples

```python
from capa import CoreAnalyzer, AnalysisConfiguration, AnalysisMode

# For mobile/web applications
fast_config = AnalysisConfiguration(mode=AnalysisMode.FAST)

# For detailed PDF reports
thorough_config = AnalysisConfiguration(mode=AnalysisMode.THOROUGH)

# For academic research
research_config = AnalysisConfiguration(mode=AnalysisMode.RESEARCH)

# For live video processing
realtime_config = AnalysisConfiguration(mode=AnalysisMode.REALTIME)
```

---

## Selective Analysis

### Enable/Disable Specific Modules

```python
from capa import AnalysisConfiguration

# Only WD analysis
wd_only = AnalysisConfiguration(
    enable_wd_analysis=True,
    enable_forehead_analysis=False,
    enable_morphology_analysis=False,
)

# Only morphology
morphology_only = AnalysisConfiguration(
    enable_wd_analysis=False,
    enable_forehead_analysis=False,
    enable_morphology_analysis=True,
)

# WD and forehead (no morphology)
partial = AnalysisConfiguration(
    enable_wd_analysis=True,
    enable_forehead_analysis=True,
    enable_morphology_analysis=False,
)
```

---

## Quality Settings

### Confidence Thresholds

```python
from capa import AnalysisConfiguration

# High confidence required
strict_config = AnalysisConfiguration(
    min_confidence_threshold=0.8,
    enable_quality_checks=True,
)

# Accept lower confidence results
lenient_config = AnalysisConfiguration(
    min_confidence_threshold=0.3,
    enable_quality_checks=False,
)
```

### Quality Check Behavior

When `enable_quality_checks=True`:
- Images are validated for minimum resolution
- Face detection confidence is verified
- Landmark quality is assessed
- Results below threshold return `None`

When `enable_quality_checks=False`:
- All detected faces are analyzed
- Lower quality results may be returned
- Confidence scores still indicate reliability

---

## Module-Specific Configuration

### WD Analyzer Options

```python
from capa.modules import WDAnalyzer

analyzer = WDAnalyzer(
    min_face_size=50,           # Minimum face size in pixels
    landmark_confidence=0.7,     # Minimum landmark detection confidence
)
```

### Forehead Analyzer Options

```python
from capa.modules import ForeheadAnalyzer

analyzer = ForeheadAnalyzer(
    angle_precision=1.0,         # Angle measurement precision (degrees)
    require_profile=False,       # Require profile view for analysis
)
```

### Morphology Analyzer Options

```python
from capa.modules import MorphologyAnalyzer

analyzer = MorphologyAnalyzer(
    include_secondary_shapes=True,   # Include secondary shape classifications
    golden_ratio_analysis=True,      # Include golden ratio measurements
)
```

---

## Multi-Angle Configuration

### Angle Specifications

```python
from capa import MultiAngleAnalyzer, AngleSpecification

# Define angle with metadata
spec = AngleSpecification(
    angle_type='frontal',
    image_path='front.jpg',
    weight=1.0,              # Weight in combined analysis
    required=True,           # Whether this angle is required
)
```

### Angle Types Reference

| Type | Description | Optimal For |
|------|-------------|-------------|
| `frontal` | Direct front view (0°) | WD, Morphology |
| `lateral_left` | Left side (90°) | Forehead |
| `lateral_right` | Right side (90°) | Forehead |
| `profile` | Full profile view | Forehead geometry |
| `three_quarter_left` | 45° left | Combined |
| `three_quarter_right` | 45° right | Combined |

### Multi-Angle Weights

```python
from capa import MultiAngleAnalyzer, AngleSpecification

specs = [
    # Higher weight for frontal (primary)
    AngleSpecification(angle_type='frontal', image_path='front.jpg', weight=2.0),

    # Standard weight for profiles
    AngleSpecification(angle_type='lateral_left', image_path='left.jpg', weight=1.0),
    AngleSpecification(angle_type='lateral_right', image_path='right.jpg', weight=1.0),
]

analyzer = MultiAngleAnalyzer()
result = analyzer.analyze_multiple_angles(angle_specs=specs, subject_id="001")
```

---

## Environment Configuration

### Logging

```python
import logging

# Enable debug logging for CAPA
logging.getLogger('capa').setLevel(logging.DEBUG)

# Or configure specific modules
logging.getLogger('capa.modules.wd').setLevel(logging.INFO)
logging.getLogger('capa.modules.forehead').setLevel(logging.WARNING)
```

### Resource Management

```python
from capa import CoreAnalyzer

# Create analyzer
analyzer = CoreAnalyzer()

# Use context manager for automatic cleanup
with CoreAnalyzer() as analyzer:
    result = analyzer.analyze_image("photo.jpg")
# Resources automatically released

# Or manual cleanup
analyzer = CoreAnalyzer()
try:
    result = analyzer.analyze_image("photo.jpg")
finally:
    analyzer.shutdown()
```

---

## Performance Tuning

### Memory Optimization

```python
from capa import AnalysisConfiguration, AnalysisMode

# For memory-constrained environments
config = AnalysisConfiguration(
    mode=AnalysisMode.FAST,
    enable_quality_checks=False,  # Reduces memory usage
)
```

### Batch Processing Optimization

```python
from capa import CoreAnalyzer

# Reuse analyzer instance for multiple images
analyzer = CoreAnalyzer()

results = []
for image_path in image_paths:
    result = analyzer.analyze_image(image_path)
    results.append(result)

# Shutdown once at the end
analyzer.shutdown()
```

### GPU Acceleration

When available, CAPA automatically uses GPU acceleration for:
- Face detection (OpenCV DNN)
- MediaPipe processing

To verify GPU usage:
```python
import cv2
print(f"CUDA available: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
```

---

## Configuration Best Practices

### 1. Start with Standard Mode

```python
# Begin with standard settings
config = AnalysisConfiguration(mode=AnalysisMode.STANDARD)

# Adjust based on results
```

### 2. Match Mode to Use Case

```python
# Real-time applications: prioritize speed
if is_realtime:
    config = AnalysisConfiguration(mode=AnalysisMode.REALTIME)

# Reports: prioritize accuracy
if generating_report:
    config = AnalysisConfiguration(mode=AnalysisMode.THOROUGH)
```

### 3. Handle Low Confidence Gracefully

```python
result = analyzer.analyze_image(image_path)

if result and result.processing_metadata.overall_confidence >= 0.7:
    # High confidence - use all results
    process_results(result)
elif result and result.processing_metadata.overall_confidence >= 0.5:
    # Medium confidence - use with warning
    process_results(result, with_warning=True)
else:
    # Low confidence - request better image
    request_better_image()
```

### 4. Always Cleanup Resources

```python
# Preferred: context manager
with CoreAnalyzer() as analyzer:
    result = analyzer.analyze_image(image_path)

# Alternative: try/finally
analyzer = CoreAnalyzer()
try:
    result = analyzer.analyze_image(image_path)
finally:
    analyzer.shutdown()
```

---

## Next Steps

- Review [Examples](examples.md) for practical usage
- Read the [API Reference](api_reference.md) for complete documentation
- Understand the [Scientific Foundation](scientific_foundation.md)
