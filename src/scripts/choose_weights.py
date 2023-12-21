
def normalize_and_adjust(values):
    total = sum(values)
    normalized = [round(value / total, 2) for value in values]
    rounding_error = 1 - sum(normalized)
    # adjust the largest value(s) by the rounding error
    if rounding_error != 0:
        max_value = max(normalized)
        indexes_of_max = [i for i, v in enumerate(normalized) if v == max_value]
        error_per_value = rounding_error / len(indexes_of_max)
        for index in indexes_of_max:
            normalized[index] += error_per_value
    return [round(i,3) for i in normalized]


alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1, 1.5, 2, 3, 4, 5, 7, 10, 20, 40, 88, 140, 200]
alphas.reverse()
utilities = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1, 1.5, 2, 3, 4, 5, 7, 10, 20, 40, 88, 140, 200]
kl_weight_unnorm = 1.0 # complexity

keep = []
alphas_keep = []
utilities_keep = []
for a in alphas:
    for u in utilities:
        norm = normalize_and_adjust([a, u, kl_weight_unnorm])
        if norm not in keep:
            print(a, u, norm)
            print("\n")
            keep.append(norm)
            alphas_keep.append(a)
            utilities_keep.append(u)

print(len(keep))
print(sorted(list(set(alphas_keep))))
print(sorted(list(set(utilities_keep))))
