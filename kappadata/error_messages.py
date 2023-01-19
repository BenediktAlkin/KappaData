def too_little_samples_for_class(class_idx, actual, expected):
    return f"class {class_idx} has only {actual} samples (requires at least {expected} samples)"


KD_MIX_WRAPPER_REQUIRES_SEED_OR_CONTEXT = (
    "KDMixWrapper requires either a context or a seed to ensure same mixing of samples and labels "
    "(NOTE: using a seed will additionally make the transform deterministc per sample)"
)
