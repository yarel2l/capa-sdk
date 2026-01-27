"""
Individual Module Usage Example

This example demonstrates how to use individual CAPA analysis modules
directly for more granular control.

Usage:
    python individual_modules.py path/to/image.jpg
"""

import cv2
import numpy as np
from capa.modules import WDAnalyzer, ForeheadAnalyzer, MorphologyAnalyzer


def analyze_with_individual_modules(image_path: str) -> None:
    """Use individual analysis modules directly."""

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    print(f"Loaded image: {image.shape[1]}x{image.shape[0]} pixels")

    # Initialize individual analyzers
    wd_analyzer = WDAnalyzer()
    forehead_analyzer = ForeheadAnalyzer()
    morphology_analyzer = MorphologyAnalyzer()

    print("\n" + "="*60)
    print("INDIVIDUAL MODULE ANALYSIS")
    print("="*60)

    # WD Analysis
    print("\n--- WD (Bizygomatic Width) Analysis ---")
    try:
        wd_result = wd_analyzer.analyze(image)
        if wd_result:
            print(f"  WD Value: {wd_result.wd_value:.3f}")
            print(f"  Bizygomatic Width: {wd_result.bizygomatic_width:.1f}px")
            print(f"  Bigonial Width: {wd_result.bigonial_width:.1f}px")
            print(f"  Classification: {wd_result.primary_classification.value}")
            print(f"  Confidence: {wd_result.measurement_confidence*100:.1f}%")
        else:
            print("  Analysis failed")
    except Exception as e:
        print(f"  Error: {e}")

    # Forehead Analysis
    print("\n--- Forehead Analysis ---")
    try:
        forehead_result = forehead_analyzer.analyze(image)
        if forehead_result:
            print(f"  Slant Angle: {forehead_result.forehead_geometry.slant_angle_degrees:.1f} degrees")
            print(f"  Forehead Height: {forehead_result.forehead_geometry.forehead_height:.1f}px")
            print(f"  Impulsiveness Level: {forehead_result.impulsiveness_level.value}")
            print(f"  Confidence: {forehead_result.measurement_confidence*100:.1f}%")
        else:
            print("  Analysis failed")
    except Exception as e:
        print(f"  Error: {e}")

    # Morphology Analysis
    print("\n--- Morphology Analysis ---")
    try:
        morphology_result = morphology_analyzer.analyze(image)
        if morphology_result:
            print(f"  Face Shape: {morphology_result.shape_classification.primary_shape.value}")
            print(f"  Facial Index: {morphology_result.facial_proportions.facial_index:.1f}")
            print(f"  W/H Ratio: {morphology_result.facial_proportions.facial_width_height_ratio:.3f}")
            print(f"  Confidence: {morphology_result.measurement_confidence*100:.1f}%")
        else:
            print("  Analysis failed")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "="*60)


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python individual_modules.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    analyze_with_individual_modules(image_path)


if __name__ == "__main__":
    main()
