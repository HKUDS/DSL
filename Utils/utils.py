def early_stopping(log_val, bst_val, es, expected_order='HR', patience=5):
    """Early Stopping for training.

    Args:
        log_val: Current metric.
        bst_val: Best metric.
        es: Number of epochs before early stop.
        expected_order: The metric to be compared.
        patience: Number of epochs with no improvement after which training will be stopped.

    Returns:
        bst_val: Best metric.
        es: Number of epochs before early stop.
        should_stop: A flag deciding whether to early stop.
    """
    assert expected_order in ['HR', 'NDCG']
    should_stop = False
    if log_val[expected_order] > bst_val[expected_order]:
        es = 0
        bst_val = log_val
    else:
        es += 1
        if es >= patience:
            should_stop = True
    return bst_val, es, should_stop