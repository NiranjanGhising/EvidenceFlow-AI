import numpy as np
import pytest

from evidenceflow.storage.vectors import VectorRecord


def test_vector_record_retains_float_vector() -> None:
    vector = np.array(
        [0.1, 0.2, 0.3],
        dtype=np.float32,
    )

    record = VectorRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        model_name="test-model",
        vector=vector,
    )

    assert record.vector.dtype == np.float32
    assert record.vector.shape == (3,)


def test_zero_vector_norm_is_zero() -> None:
    vector = np.zeros(3, dtype=np.float32)

    assert float(np.linalg.norm(vector)) == 0.0


def test_normalized_vector_has_unit_norm() -> None:
    vector = np.array(
        [3.0, 4.0],
        dtype=np.float32,
    )

    normalized = vector / np.linalg.norm(vector)

    assert float(np.linalg.norm(normalized)) == pytest.approx(
        1.0
    )