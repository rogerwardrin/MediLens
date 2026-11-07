import Levenshtein


def cer(pred, truth):
    if not truth:
        return 0.0 if not pred else 1.0
    return Levenshtein.distance(pred, truth) / len(truth)


def wer(pred, truth):
    p, t = pred.split(), truth.split()
    if not t:
        return 0.0 if not p else 1.0
    # simple edit distance on word lists
    import difflib
    sm = difflib.SequenceMatcher(None, p, t)
    return 1.0 - sm.ratio()
