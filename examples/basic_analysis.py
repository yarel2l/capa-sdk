"""
Basic CAPA Analysis Example

This example demonstrates how to perform a basic facial analysis
using the CAPA SDK.

Usage:
    python basic_analysis.py path/to/image.jpg
"""

from capa import CoreAnalyzer, AnalysisMode, AnalysisConfiguration


def analyze_image(image_path: str) -> None:
    """Perform basic facial analysis on an image."""

    # Create configuration (optional - defaults work well)
    config = AnalysisConfiguration(
        mode=AnalysisMode.STANDARD,
        enable_wd_analysis=True,
        enable_forehead_analysis=True,
        enable_morphology_analysis=True,
    )

    # Initialize the analyzer
    analyzer = CoreAnalyzer(config=config)

    try:
        # Perform analysis
        print(f"Analyzing: {image_path}")
        result = analyzer.analyze_image(image_path)

        # Check if analysis was successful
        if result is None:
            print("Analysis failed - no face detected or image unreadable")
            return

        # Print the report
        print("\n" + "="*60)
        print("ANALYSIS RESULTS")
        print("="*60)

        # WD Analysis results
        if result.wd_result:
            wd = result.wd_result
            print(f"\nWD Analysis:")
            print(f"  - WD Value: {wd.wd_value:.3f}")
            print(f"  - Classification: {wd.primary_classification.value}")
            print(f"  - Confidence: {wd.measurement_confidence*100:.1f}%")

        # Forehead Analysis results
        if result.forehead_result:
            fh = result.forehead_result
            print(f"\nForehead Analysis:")
            print(f"  - Slant Angle: {fh.forehead_geometry.slant_angle_degrees:.1f} degrees")
            print(f"  - Impulsiveness Level: {fh.impulsiveness_level.value}")
            print(f"  - Confidence: {fh.measurement_confidence*100:.1f}%")

        # Morphology Analysis results
        if result.morphology_result:
            morph = result.morphology_result
            print(f"\nMorphology Analysis:")
            print(f"  - Face Shape: {morph.shape_classification.primary_shape.value}")
            print(f"  - Facial Index: {morph.facial_proportions.facial_index:.1f}")
            print(f"  - Confidence: {morph.measurement_confidence*100:.1f}%")

        # Overall confidence
        if result.processing_metadata:
            print(f"\nOverall Confidence: {result.processing_metadata.overall_confidence*100:.1f}%")
            print(f"Processing Time: {result.processing_metadata.total_processing_time:.2f}s")

        print("\n" + "="*60)

    finally:
        # Always shutdown the analyzer to release resources
        analyzer.shutdown()


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python basic_analysis.py <image_path>")
        print("\nExample:")
        print("  python basic_analysis.py photo.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    analyze_image(image_path)


if __name__ == "__main__":
    main()
