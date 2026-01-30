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
| `result_format` | `ResultFormat` | `STRUCTURED` | Output format for results |
| `enable_wd_analysis` | `bool` | `True` | Enable WD (bizygomatic) analysis |
| `enable_forehead_analysis` | `bool` | `True` | Enable forehead/impulsiveness analysis |
| `enable_morphology_analysis` | `bool` | `True` | Enable face shape analysis |
| `enable_neoclassical_analysis` | `bool` | `True` | Enable neoclassical canons analysis |
| `enable_quality_assessment` | `bool` | `True` | Enable image quality validation |
| `enable_continuous_learning` | `bool` | `True` | Enable adaptive learning system |
| `learning_mode` | `LearningMode` | `BALANCED` | Learning mode for adaptation |
| `enable_parallel_processing` | `bool` | `True` | Enable parallel module execution |
| `max_worker_threads` | `int` | `4` | Maximum threads for parallel processing |
| `subject_age` | `int` | `None` | Optional subject age for normalization |
| `subject_gender` | `str` | `None` | Optional subject gender for normalization |
| `subject_ethnicity` | `str` | `None` | Optional subject ethnicity for normalization |

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

# Realtime mode - for video streams (NOT YET IMPLEMENTED)
AnalysisMode.REALTIME
```

### Mode Comparison

| Mode | Speed | Accuracy | Detectors | Use Case | Status |
|------|-------|----------|-----------|----------|--------|
| `FAST` | ★★★★★ | ★★★☆☆ | 1-2 | Real-time apps | ✅ Implemented |
| `STANDARD` | ★★★★☆ | ★★★★☆ | 2-3 | General use | ✅ Implemented |
| `THOROUGH` | ★★★☆☆ | ★★★★★ | All | Detailed reports | ✅ Implemented |
| `SCIENTIFIC` | ★★★☆☆ | ★★★★★ | All | Research (2D only) | ✅ Implemented |
| `RESEARCH` | ★★☆☆☆ | ★★★★★ | All | Academic studies | ✅ Implemented |
| `REALTIME` | ★★★★★ | ★★☆☆☆ | 1 | Video streams | ⚠️ Not Implemented |

> **Note:** `REALTIME` mode is reserved for future video stream processing capabilities.
> Currently it behaves identically to `STANDARD` mode. Video processing features are planned for a future release.

### Mode Examples

```python
from capa import CoreAnalyzer, AnalysisConfiguration, AnalysisMode

# For mobile/web applications
fast_config = AnalysisConfiguration(mode=AnalysisMode.FAST)

# For detailed PDF reports
thorough_config = AnalysisConfiguration(mode=AnalysisMode.THOROUGH)

# For academic research
research_config = AnalysisConfiguration(mode=AnalysisMode.RESEARCH)

# For scientific publications (2D observables only)
scientific_config = AnalysisConfiguration(mode=AnalysisMode.SCIENTIFIC)
```

---

## Mode Output Differences

Different analysis modes return different data in the results. This section explains what to expect from each mode.

### Output by Mode

| Field/Feature | STANDARD | FAST | THOROUGH | SCIENTIFIC | RESEARCH |
|---------------|:--------:|:----:|:--------:|:----------:|:--------:|
| **WD Analysis** |
| `wd_value` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `primary_classification` | ✅ | ✅ | ✅ | ❌ | ✅ |
| `personality_profile` | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Forehead Analysis** |
| `slant_angle_degrees` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `impulsiveness_level` | ✅ | ✅ | ✅ | ❌ | ✅ |
| `neuroscience_correlations` | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Morphology Analysis** |
| `face_shape` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `3d_reconstruction` | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Additional** |
| `research_disclaimers` | ❌ | ❌ | ❌ | ❌ | ✅ |
| Early abort on poor landmarks | ❌ | ✅ | ❌ | ❌ | ❌ |

### SCIENTIFIC Mode

The `SCIENTIFIC` mode is designed for peer-reviewed research and academic publications. It returns **only 2D observable measurements** without psychological or neurological interpretations.

**Included:**
- Raw geometric measurements (angles, distances, ratios)
- Confidence intervals (CI95) for measurements
- Pose validation information
- Statistical z-scores and percentiles

**Excluded:**
- Personality classifications (e.g., "highly_social", "reserved")
- Impulsiveness levels
- Neuroscience correlations
- Psychological trait predictions

```python
from capa import CoreAnalyzer, AnalysisConfiguration, AnalysisMode

config = AnalysisConfiguration(mode=AnalysisMode.SCIENTIFIC)
analyzer = CoreAnalyzer(config=config)

result = analyzer.analyze_image("photo.jpg")

# Scientific mode returns raw measurements only
if result.wd_result:
    print(f"WD Value: {result.wd_result.wd_value:.3f} cm")
    print(f"Confidence: {result.wd_result.measurement_confidence:.1%}")
    # Note: personality_profile will have default/zero values
    # Note: primary_classification should not be used in scientific context

if result.forehead_result:
    print(f"Slant Angle: {result.forehead_result.forehead_geometry.slant_angle_degrees:.1f}°")
    # Note: impulsiveness_level should not be used in scientific context

analyzer.shutdown()
```

### RESEARCH Mode

The `RESEARCH` mode includes all correlations from peer-reviewed studies **with appropriate disclaimers**.

```python
config = AnalysisConfiguration(mode=AnalysisMode.RESEARCH)
analyzer = CoreAnalyzer(config=config)

result = analyzer.analyze_image("photo.jpg")

# Research mode includes disclaimers
if hasattr(result, 'research_disclaimers') and result.research_disclaimers:
    print("Research Disclaimers:")
    for disclaimer in result.research_disclaimers:
        print(f"  - {disclaimer}")

analyzer.shutdown()
```

### FAST Mode Behavior

`FAST` mode prioritizes speed and will **abort early** if landmark detection quality is too low:

```python
config = AnalysisConfiguration(mode=AnalysisMode.FAST)
analyzer = CoreAnalyzer(config=config)

result = analyzer.analyze_image("photo.jpg")

# Check warnings for early abort
if result and result.processing_metadata.warnings:
    for warning in result.processing_metadata.warnings:
        print(f"Warning: {warning}")
        # May include: "Aborting analysis due to poor landmarks in FAST mode"

analyzer.shutdown()
```

### THOROUGH Mode Features

`THOROUGH` mode enables all features including 3D facial reconstruction:

```python
config = AnalysisConfiguration(mode=AnalysisMode.THOROUGH)
analyzer = CoreAnalyzer(config=config)

result = analyzer.analyze_image("photo.jpg")

if result.morphology_result:
    props = result.morphology_result.facial_proportions
    # 3D-derived measurements available in THOROUGH mode
    print(f"Facial Volume Estimate: {props.facial_volume_estimate:.1f}")
    print(f"Facial Surface Area: {props.facial_surface_area:.1f}")
    print(f"Facial Convexity Angle: {props.facial_convexity_angle:.1f}°")

analyzer.shutdown()
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

### Quality Assessment

```python
from capa import AnalysisConfiguration

# Enable quality assessment (default)
config_with_quality = AnalysisConfiguration(
    enable_quality_assessment=True,
)

# Disable quality assessment for faster processing
config_no_quality = AnalysisConfiguration(
    enable_quality_assessment=False,
)
```

### Quality Assessment Behavior

When `enable_quality_assessment=True`:
- Images are validated for minimum resolution
- Face detection confidence is verified
- Landmark quality is assessed
- Low quality results include warnings in metadata

When `enable_quality_assessment=False`:
- All detected faces are analyzed
- Lower quality results may be returned
- Confidence scores still indicate reliability

---

## Module-Specific Configuration

### WD Analyzer Options

```python
from capa.modules import WDAnalyzer

analyzer = WDAnalyzer(
    enable_learning=True,        # Enable continuous learning system (default: True)
)
```

### Forehead Analyzer Options

```python
from capa.modules import ForeheadAnalyzer

analyzer = ForeheadAnalyzer(
    enable_learning=True,        # Enable continuous learning system (default: True)
    enable_neuroscience=True,    # Enable neuroscience correlations (default: True)
)
```

### Morphology Analyzer Options

```python
from capa.modules import MorphologyAnalyzer

analyzer = MorphologyAnalyzer(
    enable_3d_reconstruction=True,   # Enable 3D facial reconstruction (default: True)
    enable_learning=True,            # Enable continuous learning system (default: True)
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
