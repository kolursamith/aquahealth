from src.config import CLASS_NAMES, NUM_CLASSES


def test_class_count():
    assert NUM_CLASSES == 8
    assert len(CLASS_NAMES) == NUM_CLASSES
