from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import numba

def compute_wer_tokens(
    ref_tokens: Sequence,
    hyp_tokens: Sequence,
    return_alignment: bool = False,
) -> Dict:
    """基于 token 序列计算 WER/CER（Levenshtein edit distance）。

    - token 当“词” => WER
    - token 当“字/字符” => CER

    Returns:
        dict with keys:
        - wer: (S + D + I) / max(1, len(ref_tokens))
        - num_ref_tokens / num_hyp_tokens
        - num_cor / num_sub / num_del / num_ins
        - num_err / edit_distance
        - alignment（可选，元素为 ('C'/'S'/'D'/'I', ref_tok_or_None, hyp_tok_or_None)）
    """
    ref = list(ref_tokens)
    hyp = list(hyp_tokens)

    n = len(ref)
    m = len(hyp)

    OP_NONE = 0
    OP_DIAG = 1  # match or substitution
    OP_DEL = 2
    OP_INS = 3

    ops: List[bytearray] = [bytearray(m + 1) for _ in range(n + 1)]

    dp_prev = list(range(m + 1))
    dp_cur = [0] * (m + 1)

    for j in range(1, m + 1):
        ops[0][j] = OP_INS

    for i in range(1, n + 1):
        dp_cur[0] = i
        ops[i][0] = OP_DEL

        ref_tok = ref[i - 1]
        for j in range(1, m + 1):
            hyp_tok = hyp[j - 1]
            sub_cost = 0 if ref_tok == hyp_tok else 1

            cost_diag = dp_prev[j - 1] + sub_cost
            cost_del = dp_prev[j] + 1
            cost_ins = dp_cur[j - 1] + 1

            best_cost = cost_diag
            best_op = OP_DIAG

            if cost_del < best_cost:
                best_cost = cost_del
                best_op = OP_DEL
            if cost_ins < best_cost:
                best_cost = cost_ins
                best_op = OP_INS

            dp_cur[j] = best_cost
            ops[i][j] = best_op

        dp_prev, dp_cur = dp_cur, dp_prev

    edit_distance = dp_prev[m]

    i = n
    j = m
    num_cor = 0
    num_sub = 0
    num_del = 0
    num_ins = 0

    alignment: Optional[List[Tuple[str, Optional[object], Optional[object]]]] = [] if return_alignment else None

    while i > 0 or j > 0:
        op = ops[i][j]

        if op == OP_DIAG:
            ref_tok = ref[i - 1]
            hyp_tok = hyp[j - 1]
            if ref_tok == hyp_tok:
                num_cor += 1
                if return_alignment:
                    alignment.append(("C", ref_tok, hyp_tok))
            else:
                num_sub += 1
                if return_alignment:
                    alignment.append(("S", ref_tok, hyp_tok))
            i -= 1
            j -= 1
        elif op == OP_DEL:
            num_del += 1
            if return_alignment:
                alignment.append(("D", ref[i - 1], None))
            i -= 1
        elif op == OP_INS:
            num_ins += 1
            if return_alignment:
                alignment.append(("I", None, hyp[j - 1]))
            j -= 1
        else:
            raise RuntimeError(f"Invalid backtrace op={op} at (i={i}, j={j})")

    if return_alignment:
        alignment.reverse()

    num_err = num_sub + num_del + num_ins
    denom = max(1, n)
    wer = num_err / denom

    return {
        "wer": wer,
        "num_ref_tokens": n,
        "num_hyp_tokens": m,
        "num_cor": num_cor,
        "num_sub": num_sub,
        "num_del": num_del,
        "num_ins": num_ins,
        "num_err": num_err,
        "edit_distance": edit_distance,
        "alignment": alignment,
    }


if __name__ == '__main__':
    print(compute_wer_tokens('你好世界', '你不好世界'))
