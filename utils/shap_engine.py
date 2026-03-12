"""
shap_engine.py
SHAP explainability for LSTM/BiLSTM cricket models.
Provides both LOCAL (single prediction) and GLOBAL (overall model) explanations.
"""
import shap
import numpy as np

SEQUENCE_LENGTH = 20
FEATURE_NAMES = [
    "Inning", "Batting Team", "Bowling Team",
    "Ball Number", "Current Score", "Wickets Fallen",
    "Run Rate", "Remaining Overs"
]

# ─────────────────────────────────────────────
# Internal: flatten seq input for KernelExplainer
# ─────────────────────────────────────────────
def _make_predict_fn(model):
    """Wrap model so KernelExplainer can call it with flat (N,8) arrays."""
    def predict_fn(X_flat):
        # X_flat: (N, 8)  →  model needs (N, SEQUENCE_LENGTH, 8)
        X_seq = np.repeat(X_flat[:, np.newaxis, :], SEQUENCE_LENGTH, axis=1)
        preds = model.predict(X_seq, verbose=0)
        # Return shape (N,) probabilities
        return preds.flatten()
    return predict_fn


# ─────────────────────────────────────────────
# Create explainer (call once, cache in session)
# ─────────────────────────────────────────────
def create_shap_explainer(model, X_background_flat):
    """
    Parameters
    ----------
    model            : Keras LSTM/BiLSTM model
    X_background_flat: np.ndarray shape (N_bg, 8) — scaled background samples
    Returns
    -------
    shap.KernelExplainer
    """
    predict_fn = _make_predict_fn(model)
    # Use k-means summary to keep background small (fast)
    bg = shap.kmeans(X_background_flat, min(50, len(X_background_flat)))
    explainer = shap.KernelExplainer(predict_fn, bg)
    return explainer


# ─────────────────────────────────────────────
# LOCAL explanation — one sample
# ─────────────────────────────────────────────
def local_shap_values(explainer, X_sample_flat):
    """
    Parameters
    ----------
    X_sample_flat : np.ndarray shape (1, 8) — single scaled sample
    Returns
    -------
    dict with keys:
        shap_values : np.ndarray (8,)
        base_value  : float
        feature_names: list of str
    """
    values = explainer.shap_values(X_sample_flat, nsamples=100)
    # values may be list (binary) or array
    if isinstance(values, list):
        sv = np.array(values[0]).flatten()
    else:
        sv = np.array(values).flatten()
    return {
        "shap_values": sv,
        "base_value": float(explainer.expected_value
                            if not isinstance(explainer.expected_value, (list, np.ndarray))
                            else explainer.expected_value[0]),
        "feature_names": FEATURE_NAMES
    }


# ─────────────────────────────────────────────
# GLOBAL explanation — multiple samples
# ─────────────────────────────────────────────
def global_shap_importance(explainer, X_test_flat, n_samples=30):
    """
    Parameters
    ----------
    X_test_flat : np.ndarray shape (N, 8)
    n_samples   : how many test rows to use (more = slower)
    Returns
    -------
    dict with keys:
        mean_abs_shap : np.ndarray (8,)  — global importance per feature
        all_shap_vals : np.ndarray (N, 8) — for beeswarm / summary plots
        feature_names : list of str
    """
    X_sub = X_test_flat[:n_samples]
    values = explainer.shap_values(X_sub, nsamples=100)
    if isinstance(values, list):
        sv = np.array(values[0])
    else:
        sv = np.array(values)
    # sv shape: (n_samples, 8)
    if sv.ndim == 3:
        sv = sv.reshape(sv.shape[0], -1)
    return {
        "mean_abs_shap": np.mean(np.abs(sv), axis=0),
        "all_shap_vals": sv,
        "feature_names": FEATURE_NAMES
    }
