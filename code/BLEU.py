import math
def calculate_bleu_score(reference_tokens,candidate_tokens, weights=(1,0,0,0)):

    # Compute n-grams for reference
    reference_ngrams = {}
    for i in range(len(reference_tokens)):
        for n in range(1, 5):
            if i + n <= len(reference_tokens):
                ngram = tuple(reference_tokens[i:i + n])
                reference_ngrams[ngram] = reference_ngrams.get(ngram, 0) + 1

    # Compute n-grams for candidate
    candidate_ngrams = {}
    for i in range(len(candidate_tokens)):
        for n in range(1, 5):
            if i + n <= len(candidate_tokens):
                ngram = tuple(candidate_tokens[i:i + n])
                candidate_ngrams[ngram] = candidate_ngrams.get(ngram, 0) + 1

    # Calculate clipped counts
    clipped_counts = []
    for n in range(1, 5):
        clipped_count = 0
        for ngram in candidate_ngrams:
            if ngram in reference_ngrams:
                clipped_count += min(candidate_ngrams[ngram], reference_ngrams[ngram])
        clipped_counts.append(clipped_count)

    # Calculate precision
    precision = []
    for ngram_count, clipped_count in zip(range(1, 5), clipped_counts):
        if sum(candidate_ngrams.values()) > 0:
            precision.append(clipped_count / sum(candidate_ngrams.values()))
        else:
            precision.append(0)

    # Calculate geometric mean
    precision_log_sum = sum(weights[i] * math.log(p) if p > 0 else float('-inf') for i, p in enumerate(precision))
    bp = min(1, math.exp(1 - len(reference_tokens) / len(candidate_tokens)))
    bleu = bp * math.exp(precision_log_sum)

    return bleu

