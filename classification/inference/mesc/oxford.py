def binarize(val, score):
    if score in ['E', 'C']:
        return int(val)
    elif score == 'M':
        return int(val > 1)
        # 0 "nan_label", 1 "noM" --> 0
        # 2 "yesM" --> 1
    elif score == 'S':
        return int(val > 1)
        # 0 "GGS", 1 "NoGS" --> 0
        # 2 "SGS" --> 1


def oxfordify(val, score, c1_only=True):
    if score == 'M':
        return "M1" if val >= 0.5 else "M0"
    elif score == 'E':
        return "E1" if val > 0 else "E0"
    elif score == 'S':
        return "S1" if val > 0 else "S0"
    elif score == 'C':
        if c1_only:
            return "C1" if val > 0 else "C0"
        else:
            if val >= 0.25: return "C2"
            elif val > 0: return "C1"
            else: return "C0"


def textify(x, lesion):
    return f"{lesion}{int(x)}"
