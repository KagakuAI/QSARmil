from math import comb

def kid_accuracy(true_key_indices, predicted_weights, top_n=1):
    assert len(predicted_weights) == len(true_key_indices), "Mismatched input lengths."

    hits = 0
    total = len(predicted_weights)

    for bag_weights, key_indices in zip(predicted_weights, true_key_indices):

        top_n_indices = sorted(range(len(bag_weights)), key=lambda i: bag_weights[i], reverse=True)[:top_n]

        if any(idx in top_n_indices for idx in key_indices):
            hits += 1

    return hits / total if total > 0 else 0.0

def expected_kid_accuracy(true_key_indices, bag_sizes, top_n=1):

    assert len(true_key_indices) == len(bag_sizes), "Mismatched input lengths."
    expected_hits = 0

    for key_indices, B in zip(true_key_indices, bag_sizes):
        K = len(key_indices)
        N = min(top_n, B)  # top_n can't exceed bag size

        if K == 0 or B == 0:
            continue  # skip invalid bags

        if B - K < N:
            hit_prob = 1.0  # guaranteed to pick a key instance
        else:
            hit_prob = 1 - (comb(B - K, N) / comb(B, N))

        expected_hits += hit_prob

    return expected_hits / len(true_key_indices) if true_key_indices else 0.0