def validate_input_vector(X_row, expected_features=25):
    if len(X_row[0]) != expected_features:
        return False, f"Xato: {len(X_row[0])} ta feature, lekin model {expected_features} ta kutilmoqda."
    return True, "Input to'g'ri."


def safe_predict(model, X_row):
    valid, msg = validate_input_vector(X_row)
    if not valid:
        return None, msg
    try:
        prob = model.predict_proba(X_row)[0][1]
        return prob, None
    except Exception as e:
        return None, f"Model predictda xatolik: {str(e)}"


def validate_all_models(models, X_row):
    result = {}
    for i in range(25):
        col = f"cell_{i+1}"
        model = models.get(col)
        prob, error = safe_predict(model, X_row)
        if error:
            result[col] = 0  # fallback xavfsizlik darajasi
        else:
            result[col] = prob
    return result
