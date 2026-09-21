def feature_importance(booster, importance_type="split"):
    if importance_type not in {"split", "gain"}: raise ValueError("importance_type must be 'split' or 'gain'")
    return booster.feature_importance(importance_type=importance_type).tolist()
def shap_values(booster, matrix): return booster.predict(matrix, pred_contrib=True)
