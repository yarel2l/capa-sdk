"""
CAPA SDK Test Configuration

Shared fixtures and configuration for all tests.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path


# =============================================================================
# Test Image Fixtures
# =============================================================================

@pytest.fixture
def sample_image_rgb():
    """Create a sample RGB image for testing."""
    # Create a 480x640 RGB image with a face-like region
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add a skin-tone colored oval for face region
    cv2.ellipse(image, (320, 240), (100, 130), 0, 0, 360, (180, 150, 130), -1)

    # Add eye regions
    cv2.circle(image, (280, 200), 15, (50, 50, 50), -1)
    cv2.circle(image, (360, 200), 15, (50, 50, 50), -1)

    # Add nose region
    cv2.line(image, (320, 220), (320, 270), (160, 130, 110), 3)

    # Add mouth region
    cv2.ellipse(image, (320, 300), (30, 10), 0, 0, 180, (150, 100, 100), 2)

    return image


@pytest.fixture
def sample_image_gray(sample_image_rgb):
    """Create a grayscale version of the sample image."""
    return cv2.cvtColor(sample_image_rgb, cv2.COLOR_BGR2GRAY)


@pytest.fixture
def blank_image():
    """Create a blank image with no face."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def small_image():
    """Create a very small image (edge case)."""
    return np.zeros((10, 10, 3), dtype=np.uint8)


# =============================================================================
# Mock Landmark Fixtures
# =============================================================================

@pytest.fixture
def mock_landmarks_68():
    """Create mock 68-point facial landmarks."""
    landmarks = {}

    # Jaw line (points 0-16)
    for i in range(17):
        landmarks[i] = (150 + i * 20, 350 - abs(i - 8) * 5)

    # Right eyebrow (points 17-21)
    for i in range(17, 22):
        landmarks[i] = (180 + (i - 17) * 15, 180)

    # Left eyebrow (points 22-26)
    for i in range(22, 27):
        landmarks[i] = (320 + (i - 22) * 15, 180)

    # Nose bridge (points 27-30)
    for i in range(27, 31):
        landmarks[i] = (320, 200 + (i - 27) * 20)

    # Nose bottom (points 31-35)
    for i in range(31, 36):
        landmarks[i] = (290 + (i - 31) * 15, 280)

    # Right eye (points 36-41)
    eye_r_center = (280, 210)
    for i in range(36, 42):
        angle = (i - 36) * 60
        landmarks[i] = (
            int(eye_r_center[0] + 15 * np.cos(np.radians(angle))),
            int(eye_r_center[1] + 8 * np.sin(np.radians(angle)))
        )

    # Left eye (points 42-47)
    eye_l_center = (360, 210)
    for i in range(42, 48):
        angle = (i - 42) * 60
        landmarks[i] = (
            int(eye_l_center[0] + 15 * np.cos(np.radians(angle))),
            int(eye_l_center[1] + 8 * np.sin(np.radians(angle)))
        )

    # Outer lip (points 48-59)
    for i in range(48, 60):
        angle = (i - 48) * 30
        landmarks[i] = (
            int(320 + 30 * np.cos(np.radians(angle))),
            int(320 + 15 * np.sin(np.radians(angle)))
        )

    # Inner lip (points 60-67)
    for i in range(60, 68):
        angle = (i - 60) * 45
        landmarks[i] = (
            int(320 + 15 * np.cos(np.radians(angle))),
            int(320 + 8 * np.sin(np.radians(angle)))
        )

    return landmarks


@pytest.fixture
def mock_landmarks_mediapipe():
    """Create mock MediaPipe-style landmarks (478 points as normalized coords)."""
    landmarks = []
    for i in range(478):
        landmarks.append({
            'x': 0.3 + np.random.uniform(-0.2, 0.4),
            'y': 0.3 + np.random.uniform(-0.2, 0.4),
            'z': np.random.uniform(-0.1, 0.1)
        })
    return landmarks


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def default_analysis_config():
    """Create default analysis configuration."""
    from capa import AnalysisConfiguration, AnalysisMode

    return AnalysisConfiguration(
        mode=AnalysisMode.FAST,
        enable_wd_analysis=True,
        enable_forehead_analysis=True,
        enable_morphology_analysis=True,
    )


@pytest.fixture
def minimal_analysis_config():
    """Create minimal analysis configuration (only WD)."""
    from capa import AnalysisConfiguration, AnalysisMode

    return AnalysisConfiguration(
        mode=AnalysisMode.FAST,
        enable_wd_analysis=True,
        enable_forehead_analysis=False,
        enable_morphology_analysis=False,
    )


# =============================================================================
# Test Data Paths
# =============================================================================

@pytest.fixture
def test_data_dir():
    """Get path to test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def sample_image_path(test_data_dir, sample_image_rgb):
    """Create and save a sample image, return its path."""
    test_data_dir.mkdir(exist_ok=True)
    image_path = test_data_dir / "sample_face.jpg"
    cv2.imwrite(str(image_path), sample_image_rgb)
    yield image_path
    # Cleanup
    if image_path.exists():
        image_path.unlink()


# =============================================================================
# Utility Functions
# =============================================================================

def assert_result_valid(result, expected_modules=None):
    """Helper to validate analysis result structure."""
    assert result is not None

    if expected_modules:
        if 'wd' in expected_modules:
            assert hasattr(result, 'wd_result')
        if 'forehead' in expected_modules:
            assert hasattr(result, 'forehead_result')
        if 'morphology' in expected_modules:
            assert hasattr(result, 'morphology_result')


def assert_confidence_valid(confidence):
    """Helper to validate confidence values."""
    assert 0.0 <= confidence <= 1.0
