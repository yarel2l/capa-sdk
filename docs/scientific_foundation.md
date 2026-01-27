# Scientific Foundation

CAPA SDK is built on peer-reviewed scientific research in craniofacial analysis, neuroscience, and anthropometry.

## Research Papers

All referenced papers are included in the `data/` folder of this repository.

### WD Analysis (Bizygomatic Width)

The WD (Width Difference) analysis measures the ratio between bizygomatic width and bigonial width, correlating with social personality traits.

| Paper | Key Findings |
|-------|--------------|
| **Bizygomatic Width and Personality Traits** | Correlation between facial width and relational field traits |
| **Bizygomatic Width and Social Traits in Males** | Association with social and personality characteristics |
| **Gray Matter Volume and Impulsiveness** | Neural correlates of facial measurements |

#### Scientific Basis

- **Bizygomatic Width**: Distance between the zygomatic arches (cheekbones)
- **Bigonial Width**: Distance between the gonion points (jaw angles)
- **WD Ratio**: Bizygomatic / Bigonial width ratio

#### Classification Ranges

| Classification | WD Ratio | Interpretation |
|----------------|----------|----------------|
| Highly Social | > 1.10 | Strong extroversion tendency |
| Moderately Social | 1.05 - 1.10 | Above average sociability |
| Balanced | 0.95 - 1.05 | Balanced social orientation |
| Reserved | 0.90 - 0.95 | Introverted tendency |
| Highly Reserved | < 0.90 | Strong introversion tendency |

---

### Forehead Analysis (Frontal Inclination)

The forehead analysis measures the frontal inclination angle, which correlates with impulsiveness levels based on neuroscience research.

| Paper | Key Findings |
|-------|--------------|
| **Slant of the Forehead as Feature of Impulsiveness** | Direct correlation between forehead angle and BIS-11 scores |
| **Cortical Thickness and Forehead Slant** | Neuroanatomical basis for the correlation |
| **Frontonasal Dysmorphology in Bipolar Disorder** | Clinical applications |

#### Scientific Basis

- **Slant Angle**: Angle of forehead relative to vertical
- **Neurological Correlation**: Relationship with prefrontal cortex development
- **BIS-11 Validation**: Validated against Barratt Impulsiveness Scale

#### Angle Interpretation

| Angle Range | Level | Interpretation |
|-------------|-------|----------------|
| < 10° | Very Low | Very low impulsiveness |
| 10° - 15° | Low | Low impulsiveness |
| 15° - 20° | Moderately Low | Below average |
| 20° - 25° | Moderate | Average impulsiveness |
| 25° - 30° | Moderately High | Above average |
| 30° - 35° | High | High impulsiveness |
| > 35° | Very High | Very high impulsiveness |

---

### Morphology Analysis (Face Shape)

Morphology analysis classifies face shapes and measures facial proportions based on anthropometric studies.

| Paper | Key Findings |
|-------|--------------|
| **Neoclassical Facial Canons in Turkish Adults** | Validity of classical facial proportions |
| **Face Shape Evaluation in Turkish Individuals** | Face shape classification methodology |
| **3D Anthropometric Facial Analysis** | Precision measurements |
| **Facial Index Studies (Chilean Population)** | Population-specific indices |

#### Scientific Basis

- **Facial Index**: (Facial height / Facial width) × 100
- **Prosopic Classification**: Euryprosopic, Mesoprosopic, Leptoprosopic
- **Golden Ratio**: Classical proportion analysis

#### Face Shape Classifications

| Shape | Characteristics |
|-------|-----------------|
| Oval | Balanced proportions, slightly narrow forehead |
| Round | Equal width and height, soft features |
| Square | Angular jaw, wide forehead |
| Rectangular | Long face, angular features |
| Heart | Wide forehead, narrow chin |
| Diamond | Wide cheekbones, narrow forehead and chin |
| Triangular | Narrow forehead, wide jaw |
| Oblong | Long face, balanced width |

---

### Neoclassical Canons

Analysis of classical facial proportions based on the work of Farkas and historical standards.

| Paper | Key Findings |
|-------|--------------|
| **Farkas Principles for Facial Beauty** | Validation of classical canons |
| **Anthropometry of the Head and Face** | Comprehensive measurement standards |

#### The 8 Neoclassical Canons

1. **Equal Thirds**: Face divided into equal vertical thirds
2. **Eye Width**: Eye width equals interocular distance
3. **Nose Width**: Nose width equals interocular distance
4. **Mouth Width**: Mouth width equals 1.5× nose width
5. **Nose Length**: Nose length equals ear length
6. **Interocular Distance**: Eyes one eye-width apart
7. **Face Width**: 4× nose width
8. **Lower Face**: Lower third equals nose length

---

## Measurement Methodology

### Landmark Detection

CAPA uses an ensemble of multiple detectors for robust landmark identification:

- **dlib** - 68-point facial landmarks
- **MediaPipe** - 478-point face mesh
- **face_recognition** - Face encoding and landmarks
- **OpenCV DNN** - Deep learning face detection

### Quality Assurance

- **Adaptive Quality System**: Dynamic threshold adjustment
- **Cross-Validation**: Multi-detector consensus
- **Confidence Scoring**: Reliability metrics for all measurements

### Continuous Learning

- **Performance Tracking**: Historical analysis metrics
- **Parameter Optimization**: Adaptive threshold tuning
- **Outlier Detection**: Isolation Forest for anomaly detection

---

## Limitations and Disclaimers

### Scientific Limitations

1. **2D Analysis**: All measurements are from 2D images, not 3D scans
2. **Population Variance**: Research based on specific populations
3. **Individual Variation**: Results are statistical correlations, not deterministic

### Ethical Considerations

1. **Not Diagnostic**: Results should not be used for clinical diagnosis
2. **Professional Interpretation**: Qualified professionals should interpret results
3. **No Employment Decisions**: Should not be used for hiring or employment

### Accuracy Considerations

- Results depend on image quality and face orientation
- Optimal results require frontal face images
- Multiple angles improve accuracy
- Confidence scores indicate reliability

---

## References

### Primary Papers

1. Gabarre-Mir, C., et al. "Bizygomatic Width and Personality Traits of the Relational Field"
2. Guerrero, M., et al. "The Slant of the Forehead as a Craniofacial Feature of Impulsiveness"
3. Gabarre-Armengol, C., et al. "Correlation between Impulsiveness, Cortical Thickness and Slant of The Forehead"
4. Karaman, F., et al. "The validity of eight neoclassical facial canons in the Turkish adults"
5. Arslan, E., et al. "Evaluation of Face Shape in Turkish Individuals"

### Foundational Works

- Farkas, L.G. (1994). *Anthropometry of the Head and Face*. Raven Press.
- Kolar, J.C. & Salter, E.M. (1997). *Craniofacial Anthropometry*. Charles C Thomas.
