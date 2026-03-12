"""
prediction_engine.py
Loads LSTM/BiLSTM models and provides prediction + preprocessing.
Models trained with SEQUENCE_LENGTH=20. Predictions use shape (1,20,8).
"""
import numpy as np
import joblib
import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

SEQUENCE_LENGTH = 20  # Must match train_model.py
FEATURE_NAMES = [
    "inning", "batting_team", "bowling_team",
    "ball_number", "current_score", "wickets_fallen",
    "run_rate", "remaining_overs"
]

# ─── Load models & preprocessing ──────────────────────────────────────────────
lstm_model    = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_model.h5"))
bilstm_model  = tf.keras.models.load_model(os.path.join(MODEL_DIR, "bilstm_model.h5"))
scaler        = joblib.load(os.path.join(MODEL_DIR, "feature_scaler.pkl"))
encoders      = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))


# ─── Team list ────────────────────────────────────────────────────────────────
def get_teams():
    return sorted(list(encoders["batting_team"].classes_))


# ─── Preprocess raw inputs → scaled flat array (1,8) ─────────────────────────
def preprocess_input(inning, batting_team, bowling_team, ball_number,
                     current_score, wickets_fallen, run_rate, remaining_overs):
    batting_enc  = encoders["batting_team"].transform([batting_team])[0]
    bowling_enc  = encoders["bowling_team"].transform([bowling_team])[0]

    raw = np.array([[
        inning, batting_enc, bowling_enc,
        ball_number, current_score, wickets_fallen,
        run_rate, remaining_overs
    ]], dtype=float)

    scaled = scaler.transform(raw)          # shape (1, 8)
    return scaled                           # flat — callers decide reshape


# ─── Reshape flat → LSTM sequence (1, SEQ_LEN, 8) ────────────────────────────
def to_sequence(X_flat):
    """X_flat shape (1,8)  →  (1, SEQUENCE_LENGTH, 8)"""
    return np.repeat(X_flat[:, np.newaxis, :], SEQUENCE_LENGTH, axis=1)


# ─── Predict from flat scaled input ──────────────────────────────────────────
def predict_match(X_flat):
    """
    Parameters
    ----------
    X_flat : np.ndarray shape (1, 8) — output of preprocess_input()
    Returns
    -------
    dict with prediction metadata
    """
    X_seq = to_sequence(X_flat)

    lstm_prob   = float(lstm_model.predict(X_seq, verbose=0)[0][0])
    bilstm_prob = float(bilstm_model.predict(X_seq, verbose=0)[0][0])
    win_prob    = (lstm_prob + bilstm_prob) / 2.0
    loss_prob   = 1.0 - win_prob
    prediction  = "WIN" if win_prob >= 0.5 else "LOSS"

    return {
        "prediction":       prediction,
        "win_probability":  win_prob,
        "loss_probability": loss_prob,
        "lstm_probability": lstm_prob,
        "bilstm_probability": bilstm_prob,
    }


# ─── Probability helpers ──────────────────────────────────────────────────────
def confidence_level(prob):
    if prob >= 0.85 or prob <= 0.15: return "Very High"
    if prob >= 0.70 or prob <= 0.30: return "High"
    if prob >= 0.60 or prob <= 0.40: return "Moderate"
    return "Low"

def model_agreement(lstm_p, bilstm_p):
    return "Strong ✅" if (lstm_p >= 0.5) == (bilstm_p >= 0.5) else "Weak ⚠️"
