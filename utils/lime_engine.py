"""
lime_engine.py  —  LIME explainability for LSTM/BiLSTM cricket models.
FIX: exp.local_pred may have only 1 element when model returns single sigmoid output.
"""
import lime
import lime.lime_tabular
import numpy as np

SEQUENCE_LENGTH = 20
FEATURE_NAMES = [
    "Inning", "Batting Team", "Bowling Team",
    "Ball Number", "Current Score", "Wickets Fallen",
    "Run Rate", "Remaining Overs"
]


def _make_predict_fn(model):
    """LIME passes flat (N,8); model needs (N, SEQ_LEN, 8).
    Returns (N,2) — [P(LOSS), P(WIN)] as LIME classifier needs."""
    def predict_fn(X_flat):
        X_seq = np.repeat(X_flat[:, np.newaxis, :], SEQUENCE_LENGTH, axis=1)
        preds = model.predict(X_seq, verbose=0).flatten()
        return np.column_stack([1 - preds, preds])
    return predict_fn


def create_lime_explainer(X_train_flat):
    """
    Parameters
    ----------
    X_train_flat : np.ndarray shape (N, 8)
    Returns : lime.lime_tabular.LimeTabularExplainer
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_flat,
        feature_names=FEATURE_NAMES,
        class_names=["LOSS", "WIN"],
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )
    return explainer


def local_lime_explanation(explainer, model, X_sample_flat, num_features=8):
    """
    Parameters
    ----------
    X_sample_flat : np.ndarray shape (8,) or (1,8)
    Returns dict with weights, intercept, local_pred, model_pred, feature_names, explanation
    """
    predict_fn = _make_predict_fn(model)
    sample_1d = X_sample_flat.flatten()

    exp = explainer.explain_instance(
        sample_1d,
        predict_fn,
        num_features=num_features,
        num_samples=500,
        labels=(1,)   # explain WIN class (index 1)
    )

    weights = dict(exp.as_list(label=1))

    # ── FIX: local_pred may be array of 1 or 2 elements ──
    local_pred_arr = exp.local_pred
    if hasattr(local_pred_arr, '__len__') and len(local_pred_arr) > 1:
        local_pred_val = float(local_pred_arr[1])   # WIN probability
    elif hasattr(local_pred_arr, '__len__') and len(local_pred_arr) == 1:
        local_pred_val = float(local_pred_arr[0])
    else:
        local_pred_val = float(local_pred_arr)

    intercept_val = exp.intercept.get(1, exp.intercept.get(0, 0.0))
    model_pred_val = float(predict_fn(sample_1d.reshape(1, -1))[0, 1])

    return {
        "weights":       weights,
        "intercept":     float(intercept_val),
        "local_pred":    local_pred_val,
        "model_pred":    model_pred_val,
        "feature_names": FEATURE_NAMES,
        "explanation":   exp
    }


def global_lime_importance(explainer, model, X_test_flat, n_samples=30):
    """Average |LIME weights| over n_samples → approximate global importance."""
    predict_fn = _make_predict_fn(model)
    all_weights = []
    for i in range(min(n_samples, len(X_test_flat))):
        exp = explainer.explain_instance(
            X_test_flat[i], predict_fn,
            num_features=8, num_samples=300, labels=(1,)
        )
        row = np.zeros(8)
        for feat_str, w in exp.as_list(label=1):
            for j, fname in enumerate(FEATURE_NAMES):
                if fname.lower().replace(" ", "") in feat_str.lower().replace(" ", ""):
                    row[j] = w
                    break
        all_weights.append(row)

    arr = np.array(all_weights)
    return {
        "mean_abs_weight": np.mean(np.abs(arr), axis=0),
        "all_weights":     arr,
        "feature_names":   FEATURE_NAMES
    }
