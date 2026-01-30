# CAPA SDK Documentation

**Craniofacial Analysis & Prediction Architecture**

Welcome to the CAPA SDK documentation. This SDK provides advanced craniofacial analysis capabilities based on 15+ peer-reviewed scientific papers.

## Table of Contents

1. [Getting Started](getting_started.md) - Installation and quick start guide
2. [API Reference](api_reference.md) - Complete API documentation
3. [Scientific Foundation](scientific_foundation.md) - Research papers and methodology
4. [Examples](examples.md) - Usage examples and tutorials
5. [Configuration](configuration.md) - Configuration options

## Overview

CAPA SDK provides the following analysis capabilities:

### Core Features

| Feature | Description |
|---------|-------------|
| **WD Analysis** | Bizygomatic width measurement and social orientation classification |
| **Forehead Analysis** | Frontal inclination angle and impulsiveness level assessment |
| **Morphology Analysis** | Face shape classification and facial proportion analysis |
| **Neoclassical Canons** | Classical facial proportion validation |
| **Multi-Angle Analysis** | Combined analysis from multiple image angles |

### Quick Example

```python
from capa import CoreAnalyzer

# Initialize
analyzer = CoreAnalyzer()

# Analyze
result = analyzer.analyze_image("photo.jpg")

# Access results
print(f"WD Value: {result.wd_result.wd_value}")
print(f"Face Shape: {result.morphology_result.shape_classification.primary_shape.value}")

# Cleanup
analyzer.shutdown()
```

## Architecture

```
capa/
├── analyzers/          # Analysis orchestrators
│   ├── CoreAnalyzer
│   ├── MultiAngleAnalyzer
│   └── ResultsIntegrator
│
├── modules/            # Scientific analysis modules
│   ├── WDAnalyzer
│   ├── ForeheadAnalyzer
│   ├── MorphologyAnalyzer
│   └── NeoclassicalCanonsAnalyzer
│
└── _internal/          # Internal support systems
    ├── IntelligentLandmarkSystem
    ├── AdaptiveQualitySystem
    └── ContinuousImprovementSystem
```

## Requirements

- Python 3.9+
- OpenCV
- dlib
- MediaPipe
- face-recognition
- scikit-learn
- scipy
- numpy

## License

**Dual License (Non-Commercial Free / Commercial Paid)**

| Use Type | License | Cost |
|----------|---------|------|
| Personal projects | Non-Commercial License | Free |
| Academic research | Non-Commercial License | Free |
| Educational use | Non-Commercial License | Free |
| Non-profit internal use | Non-Commercial License | Free |
| **Commercial use** | **Commercial License Required** | Contact us |

### Patent Notice

This software implements methods covered by **US Patent 10,885,309**:
*"System and method for evaluating personality using anthropometric measurement of a person's face"*

For commercial licensing inquiries, contact: yarelleyva2@gmail.com

See the full [LICENSE](../LICENSE) file for complete terms.

## Support

For issues and questions, please visit our [GitHub repository](https://github.com/yarel2l/capa-sdk).
