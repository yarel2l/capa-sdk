"""
Multi-Angle CAPA Analysis Example

This example demonstrates how to analyze multiple images of the same
individual from different angles for more comprehensive results.

Usage:
    python multi_angle_analysis.py frontal.jpg profile.jpg
"""

from capa import MultiAngleAnalyzer, AngleSpecification


def analyze_multiple_angles(image_paths: list, subject_id: str = "subject_001") -> None:
    """Perform multi-angle facial analysis."""

    # Initialize the multi-angle analyzer
    analyzer = MultiAngleAnalyzer()

    try:
        # Create angle specifications for each image
        angle_specs = []
        angle_types = ['frontal', 'lateral_left', 'lateral_right', 'profile', 'semi_frontal']

        for i, path in enumerate(image_paths):
            # Assign angle type (or use 'frontal' as default)
            angle_type = angle_types[i] if i < len(angle_types) else 'frontal'
            spec = AngleSpecification(
                angle_type=angle_type,
                image_path=path,
                weight=1.0,
                quality_threshold=0.3
            )
            angle_specs.append(spec)
            print(f"Added image: {path} as {angle_type}")

        # Perform multi-angle analysis
        print(f"\nAnalyzing {len(angle_specs)} images for subject: {subject_id}")
        result = analyzer.analyze_multiple_angles(
            angle_specs=angle_specs,
            subject_id=subject_id
        )

        # Print results
        print("\n" + "="*60)
        print("MULTI-ANGLE ANALYSIS RESULTS")
        print("="*60)

        # Individual angle results
        print(f"\nProcessed {len(result.angle_results)} angles:")
        for angle, angle_result in result.angle_results.items():
            status = "Success" if angle_result else "Failed"
            print(f"  - {angle}: {status}")

        # Combined results
        if result.combined_wd_value is not None:
            print(f"\nCombined WD Value: {result.combined_wd_value:.3f}")

        if result.combined_forehead_angle is not None:
            print(f"Combined Forehead Angle: {result.combined_forehead_angle:.1f} degrees")

        if result.combined_face_shape:
            print(f"Combined Face Shape: {result.combined_face_shape}")

        if result.combined_confidence is not None:
            print(f"Combined Confidence: {result.combined_confidence*100:.1f}%")

        print("\n" + "="*60)

    finally:
        # Always shutdown the analyzer
        analyzer.shutdown()


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python multi_angle_analysis.py <image1> [image2] [image3] ...")
        print("\nExample:")
        print("  python multi_angle_analysis.py frontal.jpg profile.jpg")
        sys.exit(1)

    image_paths = sys.argv[1:]
    analyze_multiple_angles(image_paths)


if __name__ == "__main__":
    main()
