def confidence_level(prob):

    if prob >= 0.85:
        return "Very High"

    elif prob >= 0.70:
        return "High"

    elif prob >= 0.55:
        return "Moderate"

    else:
        return "Low"


def stability_index(win_prob, loss_prob):

    return abs(win_prob - loss_prob)


def model_agreement(lstm_pred, bilstm_pred):

    if lstm_pred == bilstm_pred:
        return "Strong Agreement"
    else:
        return "Weak Agreement"