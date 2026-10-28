"""测量各样本的原始 MLM 伪困惑度，判断该信号在本语料上是否有区分力。

背景：注册进 ENGINE_SPEC 后，mlm-perplexity 在全部 6 篇样本上都给 0.00，
怀疑是绝对标定（PIVOT/SCALE）不适合学术在散文域。这里拿原始伪困惑度，
看 AI 样本与真人样本之间是否存在可用于重标定的分离。
"""
import math
import torch
import aurora_check as ac
import evasion_suite as ev


def raw_pp(text: str, max_words: int = 250):
    """返回原始伪困惑度（不做 sigmoid 映射）。"""
    model, tok = ac._load_mlm_model()
    if model is None:
        return None
    words = text.split()
    snippet = " ".join(words[:max_words])
    enc = tok(snippet, return_tensors="pt", truncation=True, max_length=512)
    ids = enc.input_ids[0]
    special = set(tok.all_special_ids)
    nlls = []
    win, step = 64, 32
    n = len(ids)
    idx = 0
    while idx < n:
        window = ids[idx:idx + win]
        if len(window) < 8:
            break
        for j, tid in enumerate(window):
            if int(tid) in special:
                continue
            masked = window.clone()
            masked[j] = tok.mask_token_id
            with torch.no_grad():
                logits = model(masked.unsqueeze(0)).logits[0, j]
            probs = torch.softmax(logits, -1)
            p = probs[tid].item()
            nlls.append(-math.log(max(p, 1e-12)))
        idx += step
    if not nlls:
        return None
    return math.exp(sum(nlls) / len(nlls))


if __name__ == "__main__":
    for name, txt in ev.SAMPLES.items():
        pp = raw_pp(txt)
        truth = "AI" if name.startswith("S") else "human"
        print(f"{name:26s} [{truth:5s}] raw_pseudo_perplexity = {pp:.2f}"
              if pp is not None else f"{name}: pp=None")
