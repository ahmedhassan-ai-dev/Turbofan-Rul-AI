
import numpy as np
import pytest

from src.models.baseline import extract_window_features


def test_feature_extraction_values():
    windows = np.array([
        [[1.0], [2.0], [3.0]]
    ], dtype=np.float32)

    masks = np.array([
        [True, True, True]
    ])

    result = extract_window_features(windows, masks)

    assert result.shape == (1, 4)

    # Order: last, mean, std, change
    np.testing.assert_allclose(
        result[0],
        [3.0, 2.0, np.sqrt(2 / 3), 2.0],
        rtol=1e-5,
    )


def test_padding_does_not_affect_features():
    windows_a = np.array([
        [[1.0], [1.0], [2.0], [4.0]]
    ])

    windows_b = np.array([
        [[999.0], [-999.0], [2.0], [4.0]]
    ])

    masks = np.array([
        [False, False, True, True]
    ])

    features_a = extract_window_features(
        windows_a, masks
    )

    features_b = extract_window_features(
        windows_b, masks
    )

    np.testing.assert_allclose(
        features_a, features_b
    )


def test_single_valid_observation():
    windows = np.array([
        [[5.0], [5.0], [5.0]]
    ])

    masks = np.array([
        [False, False, True]
    ])

    result = extract_window_features(windows, masks)

    np.testing.assert_allclose(
        result[0],
        [5.0, 5.0, 0.0, 0.0],
    )


def test_all_padding_rejected():
    windows = np.ones((1, 3, 2))
    masks = np.zeros((1, 3), dtype=bool)

    with pytest.raises(ValueError):
        extract_window_features(windows, masks)


def test_invalid_mask_shape():
    windows = np.ones((2, 3, 2))
    masks = np.ones((2, 4), dtype=bool)

    with pytest.raises(ValueError):
        extract_window_features(windows, masks)
