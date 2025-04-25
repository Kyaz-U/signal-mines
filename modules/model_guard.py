
def validate_input_shape(avg_row, expected_shape=25):
    if avg_row.shape[1] != expected_shape:
        raise ValueError(f"Xatolik: AI model {avg_row.shape[1]} feature oldi, {expected_shape} bo'lishi kerak.")
