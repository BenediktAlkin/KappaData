def too_little_samples_for_class(class_idx, actual, expected):
    return f"class {class_idx} has only {actual} samples (requires at least {expected} samples)"