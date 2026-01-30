# API Reference

Complete API documentation for the CAPA SDK.

## Core Classes

### CoreAnalyzer

The main orchestrator for comprehensive facial analysis.

```python
from capa import CoreAnalyzer
```

#### Constructor

```python
CoreAnalyzer(
    config: Optional[AnalysisConfiguration] = None,
    quality_cache_path: Optional[str] = None,
    improvement_cache_path: Optional[str] = None
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `AnalysisConfiguration` | Analysis configuration. If None, uses defaults. |
| `quality_cache_path` | `str` | Path for quality metrics cache |
| `improvement_cache_path` | `str` | Path for learning system cache |

#### Methods

##### analyze

```python
def analyze(self, image: np.ndarray) -> Optional[ComprehensiveAnalysisResult]
```

Analyze a numpy array image.

**Parameters:**
- `image`: BGR image as numpy array

**Returns:** `ComprehensiveAnalysisResult` or `None` if analysis fails

##### analyze_image

```python
def analyze_image(self, image_path: str) -> Optional[ComprehensiveAnalysisResult]
```

Analyze an image from file path.

**Parameters:**
- `image_path`: Path to image file

**Returns:** `ComprehensiveAnalysisResult` or `None` if analysis fails

##### shutdown

```python
def shutdown(self) -> None
```

Release all resources. Always call when done.

---

### MultiAngleAnalyzer

Analyzer for multiple images of the same individual.

```python
from capa import MultiAngleAnalyzer
```

#### Constructor

```python
MultiAngleAnalyzer(config: Optional[AnalysisConfiguration] = None)
```

#### Methods

##### analyze_multiple_angles

```python
def analyze_multiple_angles(
    self,
    angle_specs: List[AngleSpecification],
    subject_id: str,
    analysis_id: Optional[str] = None
) -> MultiAngleResult
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `angle_specs` | `List[AngleSpecification]` | List of angle specifications |
| `subject_id` | `str` | Unique identifier for the subject |
| `analysis_id` | `str` | Optional analysis session ID |

**Returns:** `MultiAngleResult`

##### analyze_from_paths

```python
def analyze_from_paths(
    self,
    image_paths: List[str],
    subject_id: str
) -> MultiAngleResult
```

Convenience method to analyze from file paths.

---

## Configuration

### AnalysisConfiguration

```python
from capa import AnalysisConfiguration, AnalysisMode
```

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `AnalysisMode` | `STANDARD` | Analysis mode |
| `enable_wd_analysis` | `bool` | `True` | Enable WD analysis |
| `enable_forehead_analysis` | `bool` | `True` | Enable forehead analysis |
| `enable_morphology_analysis` | `bool` | `True` | Enable morphology analysis |
| `enable_parallel_processing` | `bool` | `True` | Enable parallel execution |
| `enable_continuous_learning` | `bool` | `True` | Enable learning system |
| `min_confidence_threshold` | `float` | `0.3` | Minimum confidence threshold |

### AnalysisMode

```python
from capa import AnalysisMode
```

| Value | Description |
|-------|-------------|
| `FAST` | Quick analysis, basic modules only |
| `STANDARD` | Standard comprehensive analysis |
| `THOROUGH` | Deep analysis with all modules |
| `SCIENTIFIC` | Maximum scientific accuracy (2D observables only) |
| `RESEARCH` | Research mode with peer-reviewed correlations |
| `REALTIME` | Optimized for real-time processing |

---

## Result Types

### ComprehensiveAnalysisResult

Complete result from CoreAnalyzer.

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `wd_result` | `WDResult` | WD analysis result |
| `forehead_result` | `ForeheadResult` | Forehead analysis result |
| `morphology_result` | `MorphologyResult` | Morphology analysis result |
| `processing_metadata` | `ProcessingMetadata` | Processing information |
| `landmark_result` | `EnsembleResult` | Landmark detection result |

### WDResult

Result from WD (Width Difference) analysis.

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `wd_value` | `float` | WD difference value in centimeters (bizygomatic - bigonial) |
| `bizygomatic_width` | `float` | Bizygomatic width in pixels |
| `bigonial_width` | `float` | Bigonial width in pixels |
| `primary_classification` | `WDClassification` | Social orientation classification |
| `measurement_confidence` | `float` | Confidence score (0-1) |
| `personality_correlations` | `WDPersonalityProfile` | Personality trait correlations |

### WDClassification

```python
from capa import WDClassification
```

Based on Gabarre-Armengol et al. (2019) research:

| Value | WD Range (cm) | Description |
|-------|---------------|-------------|
| `HIGHLY_SOCIAL` | ≥ 5.0 | Strong social orientation |
| `MODERATELY_SOCIAL` | 2.0 to 5.0 | Above average sociability |
| `BALANCED_SOCIAL` | -2.0 to 2.0 | Balanced social behavior |
| `RESERVED` | -5.0 to -2.0 | Reserved personality |
| `HIGHLY_RESERVED` | < -5.0 | Introverted tendency |

### ForeheadResult

Result from forehead analysis.

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `forehead_geometry` | `ForeheadGeometry` | Geometric measurements |
| `impulsiveness_level` | `ImpulsivenessLevel` | Impulsiveness classification |
| `measurement_confidence` | `float` | Confidence score (0-1) |
| `neuroscience_correlations` | `NeuroscienceCorrelations` | Neuroscience data |

### ForeheadGeometry

| Field | Type | Description |
|-------|------|-------------|
| `slant_angle_degrees` | `float` | Forehead slant angle |
| `forehead_height` | `float` | Height in pixels |
| `forehead_width` | `float` | Width in pixels |

### ImpulsivenessLevel

| Value | Angle Range | Description |
|-------|-------------|-------------|
| `VERY_LOW` | < 10° | Very low impulsiveness |
| `LOW` | 10° - 15° | Low impulsiveness |
| `MODERATELY_LOW` | 15° - 20° | Moderately low |
| `MODERATE` | 20° - 25° | Moderate impulsiveness |
| `MODERATELY_HIGH` | 25° - 30° | Moderately high |
| `HIGH` | 30° - 35° | High impulsiveness |
| `VERY_HIGH` | > 35° | Very high impulsiveness |

### MorphologyResult

Result from morphology analysis.

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `shape_classification` | `ShapeClassificationResult` | Face shape |
| `facial_proportions` | `FacialProportions` | Proportion measurements |
| `measurement_confidence` | `float` | Confidence score (0-1) |

### FaceShape

```python
from capa import FaceShape
```

| Value | Description |
|-------|-------------|
| `OVAL` | Oval face shape |
| `ROUND` | Round face shape |
| `SQUARE` | Square face shape |
| `RECTANGULAR` | Rectangular face shape |
| `HEART` | Heart-shaped face |
| `DIAMOND` | Diamond face shape |
| `TRIANGULAR` | Triangular face shape |
| `OBLONG` | Oblong/long face shape |
| `PENTAGONAL` | Pentagonal face shape |

### FacialProportions

| Field | Type | Description |
|-------|------|-------------|
| `facial_index` | `float` | Facial height/width index |
| `facial_width_height_ratio` | `float` | Width to height ratio |
| `upper_face_ratio` | `float` | Upper face proportion |
| `middle_face_ratio` | `float` | Middle face proportion |
| `lower_face_ratio` | `float` | Lower face proportion |

---

## Scientific Modules

### WDAnalyzer

```python
from capa.modules import WDAnalyzer

analyzer = WDAnalyzer(enable_learning: bool = True)
result = analyzer.analyze(image: np.ndarray) -> Optional[WDResult]
```

### ForeheadAnalyzer

```python
from capa.modules import ForeheadAnalyzer

analyzer = ForeheadAnalyzer(enable_learning: bool = True)
result = analyzer.analyze(image: np.ndarray) -> Optional[ForeheadResult]
```

### MorphologyAnalyzer

```python
from capa.modules import MorphologyAnalyzer

analyzer = MorphologyAnalyzer(enable_learning: bool = True)
result = analyzer.analyze(image: np.ndarray) -> Optional[MorphologyResult]
```

### NeoclassicalCanonsAnalyzer

```python
from capa.modules import NeoclassicalCanonsAnalyzer

analyzer = NeoclassicalCanonsAnalyzer()
result = analyzer.analyze(image: np.ndarray) -> Optional[NeoclassicalAnalysisResult]
```

---

## Angle Specification

### AngleSpecification

```python
from capa import AngleSpecification

spec = AngleSpecification(
    angle_type: str,           # 'frontal', 'lateral_left', 'lateral_right', 'profile', 'semi_frontal'
    image_path: str,           # Path to image file
    weight: float = 1.0,       # Weight for combining results
    quality_threshold: float = 0.3  # Minimum quality threshold
)
```

### MultiAngleResult

| Field | Type | Description |
|-------|------|-------------|
| `subject_id` | `str` | Subject identifier |
| `analysis_id` | `str` | Analysis session ID |
| `timestamp` | `datetime` | Analysis timestamp |
| `angle_results` | `Dict[str, ComprehensiveAnalysisResult]` | Results per angle |
| `combined_wd_value` | `float` | Combined WD value |
| `combined_forehead_angle` | `float` | Combined forehead angle |
| `combined_face_shape` | `str` | Combined face shape |
| `combined_confidence` | `float` | Combined confidence |
