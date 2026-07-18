"""Reference solutions for Label Encoding."""

import numpy as np

def fit_label_encoder(labels):
    classes = np.array(sorted(np.unique(labels)), dtype=object)
    mapping = {label: i for i, label in enumerate(classes)}
    return classes, mapping

def transform_labels(labels, mapping):
    return np.array([mapping[label] for label in labels], dtype=int)

def inverse_transform_labels(encoded, classes):
    classes = np.asarray(classes, dtype=object)
    return classes[np.asarray(encoded, dtype=int)]

def check(c, m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    classes, mapping = fit_label_encoder(["dog", "cat", "dog"])
    check(classes.tolist() == ["cat", "dog"], "classes")
    print("PASS  classes")
    enc = transform_labels(["dog", "cat"], mapping)
    check(np.array_equal(enc, [1, 0]), "encoded labels")
    print("PASS  encoded labels")
    check(inverse_transform_labels(enc, classes).tolist() == ["dog", "cat"], "inverse labels")
    print("PASS  inverse labels")
    print("All tests passed.")
