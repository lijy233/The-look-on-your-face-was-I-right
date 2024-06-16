import math
from collections import Counter


def calculate_bleu_score(reference_tokens, candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25)):
    # 计算参考句子的n-gram频率
    reference_ngrams = [Counter(tuple(reference_tokens[i:i + n]) for i in range(len(reference_tokens) - n + 1)) for n in
                        range(1, 5)]

    # 计算候选句子的n-gram频率
    candidate_ngrams = [Counter(tuple(candidate_tokens[i:i + n]) for i in range(len(candidate_tokens) - n + 1)) for n in
                        range(1, 5)]

    # 初始化裁剪计数和总计数列表
    clipped_counts = []
    total_counts = []
    for n in range(4):  # 对于每一个n-gram (1-gram 到 4-gram)
        clipped_count = 0
        total_count = 0
        for ngram in candidate_ngrams[n]:  # 遍历候选句子的每个n-gram
            if ngram in reference_ngrams[n]:  # 如果n-gram在参考句子中
                # 裁剪计数为候选句子和参考句子中n-gram出现次数的最小值
                clipped_count += min(candidate_ngrams[n][ngram], reference_ngrams[n][ngram])
            # 总计数为候选句子中n-gram的出现次数
            total_count += candidate_ngrams[n][ngram]
        clipped_counts.append(clipped_count)
        total_counts.append(total_count)

    # 计算每个n-gram的精度
    precisions = [clipped / total if total > 0 else 0 for clipped, total in zip(clipped_counts, total_counts)]

    # 计算加权几何平均
    if all(p == 0 for p in precisions):  # 如果所有精度都是0
        precision_log_sum = float('-inf')  # 对数和设为负无穷
    else:
        precision_log_sum = sum(w * math.log(p) if p > 0 else 0 for p, w in zip(precisions, weights))

    # 计算简短惩罚
    ref_length = len(reference_tokens)
    cand_length = len(candidate_tokens)
    if cand_length > ref_length:
        bp = 1  # 如果候选句子长度大于等于参考句子，则简短惩罚为1
    else:
        bp = math.exp(1 - ref_length / cand_length)  # 否则为exp(1 - ref_length / cand_length)

    # 计算BLEU分数
    bleu = bp * math.exp(precision_log_sum)

    return bleu
