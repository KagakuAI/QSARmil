from math import comb

def kid_accuracy(true_key_inst, predicted_weights, top_n=1):
    """
    Returns:
        acc: empirical KID accuracy
        exp: expected KID accuracy (random baseline)
    """

    assert len(predicted_weights) == len(true_key_inst), "Mismatched input lengths."

    predicted_hits = 0
    expected_hits = 0
    total_bags = len(predicted_weights)

    for key_inst, bag_weights in zip(true_key_inst, predicted_weights):

        # -------------------------
        # Predicted KID accuracy
        # -------------------------
        top_n_predicted_indices = sorted(
            range(len(bag_weights)),
            key=lambda i: bag_weights[i],
            reverse=True
        )[:top_n]

        if any(key_inst[idx] == 1 for idx in top_n_predicted_indices):
            predicted_hits += 1

        # -------------------------
        # Expected KID accuracy
        # -------------------------
        bag_size = len(bag_weights)
        num_key_instances = sum(key_inst)
        num_pred_instances = min(top_n, bag_size)

        if bag_size == 0 or num_key_instances == 0:
            continue

        if bag_size - num_key_instances < num_pred_instances:
            hit_probability = 1.0
        else:
            hit_probability = 1 - (
                comb(bag_size - num_key_instances, num_pred_instances)
                / comb(bag_size, num_pred_instances)
            )

        expected_hits += hit_probability

    acc = predicted_hits / total_bags
    exp = expected_hits / total_bags

    return acc, exp