def get_balanced_indices_by_targets(targets, samples_per_class, n_classes=10):
    indices = []

    for class_label in range(n_classes):
        class_indices = (targets == f'{class_label}')
        class_indices = [i for i, val in enumerate(class_indices) if val]
        selected = class_indices[:samples_per_class]
        indices.extend(selected)
    return indices