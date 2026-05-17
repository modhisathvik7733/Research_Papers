"""0010 analysis — verbatim exp-0002 paired-CRN three-way + effect-size gate
on (smf-adam vs smf-localpc) over seeds, from the SMF eval JSONs.

  python analyze.py <OUT_DIR> [--dump]

R-2 (THE test): is smf-localpc NOT worse than smf-adam on BOTH retention
(WikiText ppl ↓, TriviaQA acc ↑) and new knowledge (MedMCQA acc ↑)?
Honest prior (pre-registered): most likely smf-localpc ≈ smf-adam (inert).
A clean *worse* = optimiser conflicts with retention even when retention is
architectural. No quality-win is pre-claimed.

The exact metric key names in their eval JSON are not verified from static
inspection; extraction is fuzzy and FAILS LOUDLY if a metric can't be
located (run with --dump to inspect and adjust the KEYS map — do not let it
silently mis-parse).
"""
import glob
import json
import math
import os
import sys

DELTA = 0.0                     # practical-effect floor; metrics are on
                                # different scales -> use std-relative gate
KEYS = {                        # (substring match, higher_is_better)
    "medmcqa": (("medmcqa",), True),
    "triviaqa": (("triviaqa", "trivia"), True),
    "wikitext_ppl": (("perplex", "ppl", "wikitext"), False),
}


def _flat(d, pre=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flat(v, f"{pre}{k}.".lower()))
    elif isinstance(d, (int, float)):
        out[pre[:-1]] = float(d)
    return out


def _find(flat, subs):
    for k, v in flat.items():
        if any(s in k for s in subs) and isinstance(v, float):
            return v
    return None


def load(out_dir):
    rows = {}                                   # (opt,seed) -> {metric:val}
    for f in glob.glob(os.path.join(out_dir, "eval_*_s*.json")):
        base = os.path.basename(f)[5:-5]        # opt_sSEED
        opt, seed = base.rsplit("_s", 1)
        flat = _flat(json.load(open(f)))
        m = {}
        for name, (subs, _) in KEYS.items():
            v = _find(flat, subs)
            if v is None:
                raise SystemExit(
                    f"FAIL: metric '{name}' not found in {base}. Run "
                    f"`python analyze.py {out_dir} --dump` and fix KEYS. "
                    f"Refusing to mis-parse (the discipline).")
            m[name] = v
        rows[(opt, int(seed))] = m
    return rows


def _sp(k, n):
    return 1.0 if n == 0 else min(
        1.0, 2.0 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2.0 ** n)


def three_way(m, s, sg, ns):
    if ns == 0 or not (s == s) or s <= 0:
        return "NOT"
    if m > s and sg >= math.ceil(0.8 * ns):
        return "SEP"
    if m <= 0.5 * s or sg < math.ceil(0.6 * ns):
        return "NOT"
    return "AMB"


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    out_dir = sys.argv[1]
    if "--dump" in sys.argv:
        for f in sorted(glob.glob(os.path.join(out_dir, "eval_*.json"))):
            print(f"\n== {f} ==\n{json.dumps(_flat(json.load(open(f))), indent=2)}")
        return
    rows = load(out_dir)
    seeds = sorted({s for (_, s) in rows})
    have = [s for s in seeds if ("adamw", s) in rows and ("localpc", s) in rows]
    if not have:
        raise SystemExit("FAIL: need paired adamw & localpc per seed.")
    print(f"0010 paired analysis — seeds {have} (n={len(have)})")
    print("smf-adam vs smf-localpc; >0 ⇒ localpc better on that metric\n")
    import numpy as np
    verdicts = {}
    for name, (_, hib) in KEYS.items():
        d = []
        for s in have:
            a = rows[("adamw", s)][name]
            l = rows[("localpc", s)][name]
            d.append((l - a) if hib else (a - l))   # >0 => localpc better
        d = np.array(d, float)
        ns = len(d)
        mm, ss = float(d.mean()), float(d.std(ddof=1)) if ns > 1 else float("nan")
        sg = int((d > 0).sum())
        # "not worse" = NOT-separated-bad OR separated-good
        worse = (mm < -ss) and sg < math.ceil(0.4 * ns)
        v = three_way(mm, ss, sg, ns)
        verdicts[name] = (not worse, v, mm, ss, sg)
        print(f"  {name:<13} Δ m={mm:+.4f} s={ss:.4f} sign={sg}/{ns} "
              f"p={_sp(sg, ns):.3f} -> {v}"
              f"{'  [LOCALPC WORSE]' if worse else ''}")
    print("-" * 64)
    notworse_all = all(v[0] for v in verdicts.values())
    print("R-2 VERDICT:",
          "NON-CONFLICT — smf-localpc not worse than smf-adam on all "
          "metrics. The two survivors COEXIST on a real SMF model "
          "(coexistence; per the pre-registered prior, most likely "
          "Local-PC ≈ Adam i.e. inert — NOT a quality win)."
          if notworse_all else
          "CONFLICT — smf-localpc worse than smf-adam on >=1 metric: the "
          "NL-style optimiser degrades retention even when retention is "
          "architectural. Hardened negative; the synthesis fails on a real "
          "model. Reported straight.")
    print("R-3 scope: deep-unroll value of Local-PC (exp-0005) NOT tested "
          "here; no quality-win pre-claimed regardless of R-2.")


if __name__ == "__main__":
    main()
