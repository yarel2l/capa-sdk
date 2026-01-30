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

##### analyze_image

```python
def analyze_image(
    self,
    image: Union[np.ndarray, str, Path],
    analysis_id: Optional[str] = None,
    subject_id: Optional[str] = None,
    angle_type: Optional[str] = None
) -> ComprehensiveAnalysisResult
```

Analyze an image (numpy array or file path).

**Parameters:**
- `image`: Input image as numpy array OR path to image file
- `analysis_id`: Optional identifier for this analysis
- `subject_id`: Optional identifier for the subject
- `angle_type`: Optional angle type ('frontal', 'lateral', 'semi_frontal', 'profile')

**Returns:** `ComprehensiveAnalysisResult`

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
| `result_format` | `ResultFormat` | `STRUCTURED` | Output format for results |
| `enable_wd_analysis` | `bool` | `True` | Enable WD analysis |
| `enable_forehead_analysis` | `bool` | `True` | Enable forehead analysis |
| `enable_morphology_analysis` | `bool` | `True` | Enable morphology analysis |
| `enable_neoclassical_analysis` | `bool` | `True` | Enable neoclassical canons analysis |
| `enable_quality_assessment` | `bool` | `True` | Enable image quality validation |
| `enable_continuous_learning` | `bool` | `True` | Enable learning system |
| `learning_mode` | `LearningMode` | `BALANCED` | Learning mode for adaptation |
| `enable_parallel_processing` | `bool` | `True` | Enable parallel execution |
| `max_worker_threads` | `int` | `4` | Maximum threads for parallel processing |
| `subject_age` | `int` | `None` | Optional subject age for normalization |
| `subject_gender` | `str` | `None` | Optional subject gender |
| `subject_ethnicity` | `str` | `None` | Optional subject ethnicity |

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
| `REALTIME` | ⚠️ **Not yet implemented** - Reserved for video streams |

> **Note:** See [Configuration - Mode Output Differences](configuration.md#mode-output-differences) for detailed information about what each mode returns.

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
| `wd_ratio` | `float` | WD ratio |
| `normalized_wd_value` | `float` | Age/gender normalized WD value |
| `confidence_weighted_wd` | `float` | Confidence weighted WD value |
| `landmark_quality` | `WDLandmarkQuality` | Landmark quality assessment |
| `measurement_confidence` | `float` | Confidence score (0-1) |
| `analysis_reliability` | `float` | Analysis reliability score |
| `personality_profile` | `WDPersonalityProfile` | Personality trait correlations |
| `primary_classification` | `WDClassification` | Social orientation classification |
| `secondary_traits` | `List[str]` | Secondary personality traits |
| `research_correlations` | `Dict[str, float]` | Research-based correlations |
| `analysis_id` | `str` | Unique analysis identifier |
| `timestamp` | `datetime` | Analysis timestamp |

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
| `slant_angle_degrees` | `float` | Forehead slant angle in degrees |
| `slant_angle_radians` | `float` | Forehead slant angle in radians |
| `forehead_height` | `float` | Height in pixels |
| `forehead_width` | `float` | Width in pixels |
| `forehead_curvature` | `float` | Curvature measurement |
| `frontal_prominence` | `float` | Frontal bone prominence |
| `temporal_width` | `float` | Temporal region width |
| `hairline_recession` | `float` | Hairline recession measurement |
| `nasion_sellion_angle` | `float` | Nasion-sellion angle |
| `frontal_bone_angle` | `float` | Frontal bone angle |
| `supraorbital_angle` | `float` | Supraorbital angle |
| `forehead_face_ratio` | `float` | Forehead to face ratio |
| `width_height_ratio` | `float` | Forehead width to height ratio |
| `curvature_angle_ratio` | `float` | Curvature to angle ratio |

### ImpulsivenessLevel

| Value | Angle Range | Description |
|-------|-------------|-------------|
| `VERY_LOW` | < 10° | Very low impulsiveness |
| `LOW` | 10° - 15° | Low impulsiveness |
| `MODERATE_LOW` | 15° - 20° | Moderately low |
| `MODERATE` | 20° - 25° | Moderate impulsiveness |
| `MODERATE_HIGH` | 25° - 30° | Moderately high |
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
| `facial_width_height_ratio` | `float` | Width to height ratio |
| `upper_face_ratio` | `float` | Upper face proportion |
| `middle_face_ratio` | `float` | Middle face proportion |
| `lower_face_ratio` | `float` | Lower face proportion |
| `bizygomatic_width` | `float` | Bizygomatic (cheekbone) width |
| `bigonial_width` | `float` | Bigonial (jaw) width |
| `temporal_width` | `float` | Temporal region width |
| `nasal_width` | `float` | Nasal width |
| `mouth_width` | `float` | Mouth width |
| `total_face_height` | `float` | Total facial height |
| `upper_face_height` | `float` | Upper face height |
| `middle_face_height` | `float` | Middle face height |
| `lower_face_height` | `float` | Lower face height |
| `forehead_height` | `float` | Forehead height |
| `facial_index` | `float` | Facial height/width index |
| `facial_cone_index` | `float` | Facial cone index |
| `nasal_index` | `float` | Nasal index |
| `oral_index` | `float` | Oral index |
| `orbital_index` | `float` | Orbital index |
| `facial_volume_estimate` | `float` | Estimated facial volume (3D) |
| `facial_surface_area` | `float` | Facial surface area (3D) |
| `facial_convexity_angle` | `float` | Facial convexity angle |
| `profile_angle` | `float` | Profile angle |

---

## Scientific Modules

### WDAnalyzer

```python
from capa.modules import WDAnalyzer

analyzer = WDAnalyzer(enable_learning: bool = True)
result = analyzer.analyze_image(image: np.ndarray) -> Optional[WDResult]
```

### ForeheadAnalyzer

```python
from capa.modules import ForeheadAnalyzer

analyzer = ForeheadAnalyzer(
    enable_learning: bool = True,
    enable_neuroscience: bool = True
)
result = analyzer.analyze_image(image: np.ndarray) -> Optional[ForeheadResult]
```

### MorphologyAnalyzer

```python
from capa.modules import MorphologyAnalyzer

analyzer = MorphologyAnalyzer(
    enable_3d_reconstruction: bool = True,
    enable_learning: bool = True
)
result = analyzer.analyze_image(image: np.ndarray) -> Optional[MorphologyResult]
```

### NeoclassicalCanonsAnalyzer

```python
from capa.modules import NeoclassicalCanonsAnalyzer

analyzer = NeoclassicalCanonsAnalyzer()
result = analyzer.analyze_image(image: np.ndarray) -> Optional[NeoclassicalAnalysisResult]
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
