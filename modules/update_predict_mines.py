
from model_guard import validate_input_shape
from logger import log_event

def predict_safest_cells(data, top_k=6):
    try:
        if "bombs_count" in data.columns:
            avg_row = data.tail(5).drop("bombs_count", axis=1).mean().values.reshape(1, -1)
        else:
            avg_row = data.tail(5).mean().values.reshape(1, -1)

        validate_input_shape(avg_row)

        predictions = {}
        for i in range(25):
            col = f"cell_{i+1}"
            prob = models[col].predict_proba(avg_row)[0][1]
            predictions[col] = prob

        safest = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [cell for cell, _ in safest]

    except Exception as e:
        log_event(f"❌ Signal xatoligi: {str(e)}")
        return ["Signal chiqarib bo'lmadi. Sabab: " + str(e)]
