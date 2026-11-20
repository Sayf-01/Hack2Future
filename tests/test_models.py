"""The saved model bundles must load and must match what the docs claim.

Every performance number stated in the README is checked here against the
artifact it came from, so documentation cannot drift away from the models.
"""
import joblib
import pytest

EXPECTED = {
    "anticancer": {
        "estimator": "RandomForestClassifier",
        "features": "aac+dpc+phys",
        "n_columns": 438,
        "threshold": 0.5,
        "cv_accuracy": 0.7386,
    },
    "hemolytic": {
        "estimator": "LGBMClassifier",
        "features": "aac+phys",
        "n_columns": 38,
        "threshold": 0.5,
        "cv_accuracy": 0.9369,
    },
}


@pytest.fixture(scope="module", params=sorted(EXPECTED))
def bundle(request, root):
    """Each saved model bundle, one at a time."""
    target = request.param
    path = root / "models" / target / "best_model.joblib"
    assert path.exists(), f"missing model artifact: {path}"
    return target, joblib.load(path)


def test_bundle_has_the_contract_keys(bundle):
    """predict.py relies on exactly these keys being present."""
    _, b = bundle
    for key in ("model", "features", "columns", "threshold"):
        assert key in b, f"bundle is missing '{key}'"


def test_estimator_matches_documentation(bundle):
    target, b = bundle
    assert type(b["model"]).__name__ == EXPECTED[target]["estimator"]


def test_feature_set_matches_documentation(bundle):
    target, b = bundle
    assert b["features"] == EXPECTED[target]["features"]
    assert len(b["columns"]) == EXPECTED[target]["n_columns"]


def test_threshold_is_the_documented_half(bundle):
    """Both thresholds were deliberately left at 0.5 - see docs/methodology.md."""
    target, b = bundle
    assert b["threshold"] == pytest.approx(EXPECTED[target]["threshold"])


def test_reported_accuracy_matches_the_artifact(bundle):
    """The README quotes these numbers; they must come from the bundle itself."""
    target, b = bundle
    assert b["cv_accuracy"] == pytest.approx(EXPECTED[target]["cv_accuracy"], abs=5e-5)


def test_model_predicts_probabilities_in_range(bundle):
    """predict_proba must give usable probabilities on a correctly shaped matrix."""
    import numpy as np
    _, b = bundle
    X = np.zeros((3, len(b["columns"])), dtype=np.float32)
    p = b["model"].predict_proba(X)[:, 1]
    assert p.shape == (3,)
    assert ((p >= 0) & (p <= 1)).all()


def test_column_names_are_unique(bundle):
    """Duplicate column names would silently misalign the feature matrix."""
    _, b = bundle
    assert len(set(b["columns"])) == len(b["columns"])
