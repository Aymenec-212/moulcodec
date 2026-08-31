#!/usr/bin/env python3
"""Structured evaluation of the Darija NeuTTS fine-tune.

More than "more samples". Four things listening alone cannot give you:

  1. CATEGORIES  - a stress suite where each group isolates one hypothesis, so a
     failure points at a cause instead of a vibe.
  2. A CEILING   - dataset rows carry both text and real tokens, so decoding the real
     tokens gives ground-truth resynthesis. That separates model error from codec error.
  3. A BASELINE  - the same text through un-finetuned neutts-air. Without it you cannot
     say the fine-tune did anything.
  4. DURATION    - an empirical speech-tokens-per-phoneme-token rate fitted from your own
     prepared data, so "too fast / too slow / truncated" becomes a number.

    python scripts/11_eval_tts.py --model Tilas/neutts-air-darija-v1
    python scripts/11_eval_tts.py --model ... --compare-base --ground-truth 6
    python scripts/11_eval_tts.py --model ... --categories qaf,msa_vs_darija --seeds 3

Writes wavs, metrics.csv, and listen.html (side-by-side players) to --out.
"""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.darija_g2p import DarijaPhonemizer, to_phones  # noqa: E402

# Reuse the prompt/device helpers rather than duplicating them. Loading by path because
# "10_generate_samples" is not a valid module name; its main() is __main__-guarded.
_spec = importlib.util.spec_from_file_location(
    "_gen10", Path(__file__).resolve().parent / "10_generate_samples.py"
)
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)
build_prompt, decode_to_wav = _gen.build_prompt, _gen.decode_to_wav
pick_device, logits_are_finite = _gen.pick_device, _gen.logits_are_finite

BASE_MODEL = "neuphonic/neutts-air"
REPO_DATA = "Tilas/MoulSot-Tokens-v1"

# Each category isolates one hypothesis about where the pipeline breaks.
SUITE: dict[str, tuple[str, list[str]]] = {
    "clusters": (
        "espeak drops short vowels, producing consonant runs. Does the model insert "
        "the schwa it heard in 95 h of audio, or read the cluster literally?",
        [
            "كتدير الخدمة ديالك بزربة",
            "نمشيو نشربو أتاي فالكافي",
            "كنقلب على شي حاجة",
        ],
    ),
    "msa_vs_darija": (
        "espeak returns the MSA form (هنا -> hunaː, not /hna/). Does the model say what "
        "it was told, or what it heard?",
        [
            "هنا كاين بزاف ديال الحركة",
            "اليوم الجو زوين بزاف",
            "دابا غادي نبدا الخدمة",
        ],
    ),
    "qaf": (
        "ق is /g/ in much of Darija but espeak always gives /q/.",
        [
            "قال ليا باللي غادي يجي",
            "كايقولو هاد الشي ماشي صحيح",
            "بغيت قهوة بلا سكر",
        ],
    ),
    "emphatics": (
        "ص ض ط ع ح خ غ - the phonemes furthest from the base model's English inventory.",
        [
            "الله يعطيك الصحة على هاد الخدمة",
            "عيني كتضربني من الصباح",
            "غادي نخدم فالطريق ديال الشرق",
        ],
    ),
    "gemination": (
        "Harakat are stripped, so shadda survives only through orthography.",
        [
            "ربي يسهل عليك هاد الشي",
            "الشمس حامية بزاف اليوم",
        ],
    ),
    "french_loans": (
        "language_switch='remove-flags' phonemises Latin script with English rules.",
        [
            "الطوموبيل ديالي عند الميكانيسيان",
            "عندي رونديفو مع الطبيب",
        ],
    ),
    "short": (
        "One or two words. min_new_tokens=50 forces >=1.0 s - watch for padding "
        "artefacts or trailing noise.",
        ["سير", "واخا", "بسلامة"],
    ),
    "long": (
        "Well past the p95 training length. Watch for drift, repetition, early stop.",
        [
            "كنظن باللي هاد الموضوع خاصو تفكير بزاف قبل ما نديرو شي قرار مستعجل فهاد الوقت",
            "ملي كنت صغير كنت كنمشي للبحر مع العائلة ديالي كل صيف ونبقاو تما شي جمعة كاملة",
        ],
    ),
    "prosody": (
        "Questions and commas. The punctuation mapping exists so these survive espeak.",
        [
            "فين غادي؟",
            "شحال هادي؟ ما شفتكش من مدة!",
            "واخا، صافي، غدا نتلاقاو.",
        ],
    ),
    "longform": (
        "~25-35 words, targeting ~10 s (500 speech tokens = your corpus p90). Long-form "
        "is where autoregressive TTS degenerates: loops, drift, stuck decoders, or "
        "stopping early. Judged on the degeneracy metrics, not just by ear.",
        [
            "اليوم صباح مشيت للسوق باش نشري شي خضرة وفواكه، ولقيت الأثمنة غالية بزاف، ولكن مع ذلك شريت شوية ديال الطماطم والبصلة والبطاطا",
            "ملي كنت صغير كنت كنمشي للمدرسة على رجلي كل صباح، وكانت الطريق طويلة شوية، ولكن كنا فرحانين حيت كنا كنلعبو مع الأصحاب فالطريق",
            "البارح تلاقيت مع صاحبي القديم اللي ما شفتوش من عشر سنين، وقعدنا كنهضرو على الذكريات ديال الطفولة حتى تشا الليل ونسينا الوقت",
            "الخدمة ديالي كتطلب مني نسافر بزاف بين المدن، وأحيانا كنبقى بعيد على الدار جمعة كاملة، ولكن كنحاول ديما نرجع فنهاية الأسبوع باش نشوف العائلة",
            "كاين بزاف ديال الناس اللي كايظنو بلي تعلم اللغات صعيب، ولكن الحقيقة هي أنه غير خاصك الصبر والممارسة اليومية، ومن بعد كل شي غادي يولي ساهل",
            "فالصيف كنمشيو للشمال حيت الجو تما زوين وباردة شوية، والبحر ديال طنجة والحسيمة كايكون صافي، والناس تما مرحبين وكريمين بزاف",
            "الطبخ المغربي مشهور فالعالم كامل، وخاصة الطاجين والكسكس اللي كنا كناكلوه كل جمعة مع العائلة، والحلويات ديال المناسبات كيف كعب الغزال",
            "مع تطور التكنولوجيا ولات الحياة ساهلة أكثر، ولكن فنفس الوقت الناس ولاو ما كايهضروش مع بعضياتهم بحال بكري، وكل واحد غارق فالتيليفون ديالو",
            "كنتمنى نتعلم شي حرفة جديدة هاد العام، يمكن النجارة ولا الطبخ ولا شي حاجة تقدر تنفعني من بعد، حيت كنشوف بلي المهارات اليدوية مهمة بزاف",
            "المشكل الكبير فالمدن الكبيرة هو الزحام ديال السيارات، خاصة فالصباح ملي الناس كايمشيو للخدمة، وكتضيع ساعة ولا ساعتين غير فالطريق",
        ],
    ),
    "common": (
        "High-frequency phrases. Should be the best-learned material in the corpus.",
        ["السلام عليكم، كيف داير؟", "الله يخليك، عافاك", "بزاف ديال الشكر ليك"],
    ),
}


# ======================================================================================
# Empirical duration model, fitted from the prepared data (no extra compute)
# ======================================================================================


def degeneracy(codes, n: int = 12):
    """Signals that generation stopped tracking the text.

    unique_ratio  distinct codes / total. Real speech at 50 Hz over a 65,536 codebook
                  barely repeats; a collapsed model repeats heavily.
    repeat_frac   fraction of positions inside a repeated n-gram - catches loops.
    max_run       longest run of one identical code - catches a stuck decoder.
    """
    if not codes:
        return 0.0, 0.0, 0
    unique = len(set(codes)) / len(codes)

    covered, seen = set(), {}
    if len(codes) >= 2 * n:
        for i in range(len(codes) - n + 1):
            g = tuple(codes[i:i + n])
            if g in seen:
                covered.update(range(seen[g], seen[g] + n))
                covered.update(range(i, i + n))
            else:
                seen[g] = i
    repeat = len(covered) / len(codes)

    run = best = 1
    for a, b in zip(codes, codes[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return unique, repeat, best


def fit_corpus_reference(data_dir: Path, head_overhead: int = 13, n_sample: int = 2000):
    """Reference bands from your own real tokens - the only honest calibration.

    In the prepared schema, speech_start_idx = len(prefix + [start] + phones + [end] +
    infix), so phoneme tokens = speech_start_idx - head_overhead and the speech ids are
    input_ids[start+1:-1]. Degeneracy is offset-invariant, so raw ids are fine.
    """
    from datasets import load_from_disk

    ds = load_from_disk(str(data_dir))["train"]
    step = max(1, len(ds) // n_sample)
    sub = ds.select([i * step for i in range(min(n_sample, len(ds)))])

    rates, uniques, repeats, runs = [], [], [], []
    for row in sub:
        ids, s = row["input_ids"], row["speech_start_idx"]
        n_phone, codes = s - head_overhead, ids[s + 1:-1]
        if n_phone > 5 and len(codes) > 25:
            rates.append(len(codes) / n_phone)
            u, rp, mr = degeneracy(codes)
            uniques.append(u)
            repeats.append(rp)
            runs.append(mr)

    def band(xs, lo=0.05, hi=0.95):
        xs = sorted(xs)
        q = lambda p: xs[int(p * (len(xs) - 1))]  # noqa: E731
        return {"p05": q(lo), "median": q(0.5), "p95": q(hi)}

    ref = {"rate": band(rates), "unique": band(uniques),
           "repeat": band(repeats), "run": band(runs), "n": len(rates)}
    print(f"corpus reference from {ref['n']:,} real clips:")
    print(f"  speech tokens per phoneme token   median {ref['rate']['median']:.2f}  "
          f"[{ref['rate']['p05']:.2f}, {ref['rate']['p95']:.2f}]")
    print(f"  unique code ratio                 median {ref['unique']['median']:.3f}  "
          f"[{ref['unique']['p05']:.3f}, {ref['unique']['p95']:.3f}]")
    print(f"  repeated 12-gram fraction         median {ref['repeat']['median']:.3f}  "
          f"p95 {ref['repeat']['p95']:.3f}")
    print(f"  longest identical-code run        median {ref['run']['median']}  "
          f"p95 {ref['run']['p95']}")
    return ref


def duration_verdict(n_phone: int, n_speech: int, ref: dict) -> tuple[float, str]:
    if n_phone <= 0:
        return float("nan"), "?"
    rate = ref["rate"]
    obs = n_speech / n_phone
    ratio = obs / rate["median"]
    if obs < rate["p05"]:
        return ratio, "TOO_SHORT"
    if obs > rate["p95"]:
        return ratio, "TOO_LONG"
    return ratio, "ok"


def degeneracy_verdict(codes, ref: dict) -> tuple[float, float, int, str]:
    """Compare against real speech rather than an invented threshold."""
    u, rp, mr = degeneracy(codes)
    flags = []
    if u < ref["unique"]["p05"]:
        flags.append("LOW_DIVERSITY")
    if rp > max(ref["repeat"]["p95"], 0.02):
        flags.append("LOOPING")
    if mr > max(ref["run"]["p95"], 3):
        flags.append("STUCK")
    return u, rp, mr, ";".join(flags) or "ok"


# ======================================================================================


def load_model(repo, device, dtype, tok):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype).to(device)
    model.config.use_cache = True
    model.eval()
    probe = tok.encode("user: Convert the text to speech:", add_special_tokens=False)
    if not logits_are_finite(model, probe, device):
        print(f"  non-finite logits in {dtype}; reloading {repo} in float32")
        del model
        if device == "mps":
            torch.mps.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).to(device)
        model.config.use_cache = True
        model.eval()
    return model


def generate(model, tok, g2p, text, device, args, seed):
    torch.manual_seed(seed)
    ids, base, phones = build_prompt(tok, g2p, text)
    speech_end = tok.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            inp,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=speech_end,
            pad_token_id=tok.pad_token_id,
        )
    codes, hit_eos = [], False
    for t in out[0, len(ids):].tolist():
        if t == speech_end:
            hit_eos = True
            break
        if base <= t < base + 65536:
            codes.append(t - base)
    n_phone = len(tok.encode(phones, add_special_tokens=False))
    return codes, phones, n_phone, hit_eos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, default=Path("eval"))
    ap.add_argument("--data", type=Path, default=Path("data/neutts_darija"),
                    help="prepared data, used only to fit the duration rate model")
    ap.add_argument("--categories", default=None, help="comma-separated subset")
    ap.add_argument("--seeds", type=int, default=1, help=">1 measures stability")
    ap.add_argument("--compare-base", action="store_true",
                    help=f"also generate with {BASE_MODEL} (another ~3 GB download)")
    ap.add_argument("--ground-truth", type=int, default=0,
                    help="N dataset rows: decode real tokens AND generate the same text")
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    ap.add_argument("--min-new-tokens", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default=None)
    args = ap.parse_args()

    import soundfile as sf
    from neucodec import NeuCodec
    from transformers import AutoTokenizer

    device, dtype = pick_device(args.device, args.dtype)
    codec_device = device if device == "cuda" else "cpu"
    print(f"device: {device} ({dtype}), codec on {codec_device}\n")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "wav").mkdir(exist_ok=True)

    ref = fit_corpus_reference(args.data) if args.data.exists() else None
    if ref is None:
        print("[warn] prepared data not found; duration and degeneracy verdicts disabled")

    tok = AutoTokenizer.from_pretrained(args.model)
    g2p = DarijaPhonemizer()
    codec = NeuCodec.from_pretrained("neuphonic/neucodec").eval().to(codec_device)

    # Build the item list -------------------------------------------------------
    wanted = set(args.categories.split(",")) if args.categories else set(SUITE)
    items = [{"category": cat, "text": t}
             for cat, (_, texts) in SUITE.items() if cat in wanted for t in texts]

    if args.ground_truth:
        from datasets import load_dataset

        ds = load_dataset(REPO_DATA, split="train")
        # Span the length range rather than sampling uniformly: without a long real clip
        # there is nothing to compare a 10 s generation against.
        probe = ds.select([i * (len(ds) // 2000) for i in range(2000)])
        order = sorted(range(len(probe)), key=lambda k: len(probe[k]["tokens"]))
        picks = [order[int(p * (len(order) - 1))]
                 for p in [j / max(1, args.ground_truth - 1) for j in range(args.ground_truth)]]
        for k, pi in enumerate(picks):
            row = probe[pi]
            codes = row["tokens"]
            path = args.out / "wav" / f"gt{k:02d}_truth.wav"
            sf.write(path, decode_to_wav(codec, codes, codec_device), 24000)
            u, rp, mr = degeneracy(codes)
            print(f"  ground truth {k}: {len(codes) / 50:5.1f} s  unique {u:.3f}  "
                  f"repeat {rp:.3f}  run {mr}")
            items.append({"category": "ground_truth", "text": row["text"],
                          "truth_wav": path.name, "truth_codes": len(codes)})
        print()

    print(f"{len(items)} items x {args.seeds} seed(s)\n")
    rows = []

    # Fine-tuned model, then base - sequentially, so only one is resident at a time.
    for tag, repo in [("ft", args.model)] + ([("base", BASE_MODEL)] if args.compare_base else []):
        print(f"--- {tag}: {repo} ---")
        model = load_model(repo, device, dtype, tok)
        for i, item in enumerate(items):
            durs = []
            for s in range(args.seeds):
                codes, phones, n_phone, hit_eos = generate(
                    model, tok, g2p, item["text"], device, args, seed=1337 + s
                )
                if len(codes) < 25:
                    print(f"  [{i}] {tag} seed{s}: only {len(codes)} codes, skipped")
                    continue
                name = f"{i:02d}_{item['category']}_{tag}_s{s}.wav"
                sf.write(args.out / "wav" / name,
                         decode_to_wav(codec, codes, codec_device), 24000)
                durs.append(len(codes) / 50)
                if ref:
                    ratio, dverdict = duration_verdict(n_phone, len(codes), ref)
                    uniq, rep, run, gverdict = degeneracy_verdict(codes, ref)
                else:
                    ratio, dverdict = float("nan"), "?"
                    uniq, rep, run = degeneracy(codes)
                    gverdict = "?"
                rows.append({
                    "idx": i, "category": item["category"], "model": tag, "seed": s,
                    "text": item["text"], "phones": phones, "n_phone_tokens": n_phone,
                    "n_codes": len(codes), "duration_s": round(len(codes) / 50, 2),
                    "hit_eos": hit_eos, "at_min_floor": len(codes) <= args.min_new_tokens + 5,
                    "at_max_tokens": len(codes) >= args.max_new_tokens - 5,
                    "rate_ratio": round(ratio, 2), "duration_verdict": dverdict,
                    "unique_ratio": round(uniq, 3), "repeat_frac": round(rep, 3),
                    "max_run": run, "degeneracy_verdict": gverdict,
                    "wav": name,
                })
                if gverdict != "ok":
                    print(f"       seed{s}: {gverdict}  unique {uniq:.3f} "
                          f"repeat {rep:.3f} run {run}")
            if durs:
                spread = (max(durs) - min(durs)) if len(durs) > 1 else 0.0
                flag = "  <-- unstable" if spread > 0.4 * statistics.mean(durs) else ""
                print(f"  [{i:2d}] {item['category']:<14} {statistics.mean(durs):.1f}s"
                      f"{f' +/-{spread / 2:.1f}' if len(durs) > 1 else ''}{flag}  "
                      f"{item['text'][:40]}")
        del model
        if device == "mps":
            torch.mps.empty_cache()

    # Outputs -------------------------------------------------------------------
    with (args.out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    write_html(args.out, items, rows, args)

    # Summary -------------------------------------------------------------------
    print("\n" + "=" * 70 + "\nSUMMARY\n" + "=" * 70)
    for tag in dict.fromkeys(r["model"] for r in rows):
        sub_all = [r for r in rows if r["model"] == tag]
        print(f"\n{tag}:")
        for cat in dict.fromkeys(r["category"] for r in sub_all):
            sub = [r for r in sub_all if r["category"] == cat]
            notes = []
            if (n := sum(r["duration_verdict"] not in ("ok", "?") for r in sub)):
                notes.append(f"{n} duration")
            if (n := sum(r["degeneracy_verdict"] not in ("ok", "?") for r in sub)):
                notes.append(f"{n} degenerate")
            if (n := sum(r["at_min_floor"] for r in sub)):
                notes.append(f"{n} at min floor")
            if (n := sum(r["at_max_tokens"] for r in sub)):
                notes.append(f"{n} hit max_new_tokens")
            if (n := sum(not r["hit_eos"] for r in sub)):
                notes.append(f"{n} no EOS")
            mean_u = statistics.mean(r["unique_ratio"] for r in sub)
            print(f"  {cat:<15} {len(sub):>3} samples  unique {mean_u:.3f}   "
                  f"{'; '.join(notes) or 'clean'}")

    if any(r["model"] == "base" for r in rows):
        print("\nfine-tune vs base (mean unique-code ratio; higher tracks real speech):")
        for cat in dict.fromkeys(r["category"] for r in rows):
            f = [r["unique_ratio"] for r in rows if r["model"] == "ft" and r["category"] == cat]
            b = [r["unique_ratio"] for r in rows if r["model"] == "base" and r["category"] == cat]
            if f and b:
                print(f"  {cat:<15} ft {statistics.mean(f):.3f}   base {statistics.mean(b):.3f}")

    print(f"\nwrote {args.out}/metrics.csv and {args.out}/listen.html")
    print("Open listen.html and compare rows within a category, not across categories.")
    return 0


def write_html(out: Path, items, rows, args) -> None:
    by_item: dict[int, list] = {}
    for r in rows:
        by_item.setdefault(r["idx"], []).append(r)

    parts = ["<!doctype html><meta charset=utf-8><title>Darija TTS eval</title>",
             "<style>body{font-family:system-ui;margin:2rem;max-width:1100px}"
             "table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;"
             "padding:.5rem;vertical-align:top;font-size:14px}"
             ".ar{font-size:18px;direction:rtl;text-align:right}"
             ".ph{color:#666;font-family:ui-monospace;font-size:12px}"
             "h2{margin-top:2rem;border-bottom:2px solid #333}"
             ".bad{color:#b00;font-weight:600}.desc{color:#555;font-style:italic}"
             "audio{width:210px}</style>",
             f"<h1>Darija TTS evaluation</h1><p class=desc>{html.escape(args.model)}</p>"]

    for cat, (desc, _) in SUITE.items():
        cat_rows = [r for r in rows if r["category"] == cat]
        if not cat_rows:
            continue
        parts.append(f"<h2>{cat}</h2><p class=desc>{html.escape(desc)}</p><table>")
        parts.append("<tr><th>text</th><th>audio</th><th>dur</th><th>check</th></tr>")
        for idx in dict.fromkeys(r["idx"] for r in cat_rows):
            group = by_item[idx]
            first = group[0]
            players = "".join(
                f"<div>{r['model']}/s{r['seed']}<br>"
                f"<audio controls src='wav/{r['wav']}'></audio></div>" for r in group
            )
            v = first["duration_verdict"]
            g = first["degeneracy_verdict"]
            check = f"<span class={'bad' if v not in ('ok', '?') else ''}>{v}</span>"
            if g not in ("ok", "?"):
                check += f" <span class=bad>{html.escape(g)}</span>"
            if first["at_min_floor"]:
                check += " <span class=bad>at floor</span>"
            if first.get("at_max_tokens"):
                check += " <span class=bad>hit max</span>"
            if not first["hit_eos"]:
                check += " <span class=bad>no EOS</span>"
            check += (f"<div class=ph>uniq {first['unique_ratio']} "
                      f"rep {first['repeat_frac']} run {first['max_run']}</div>")
            parts.append(
                f"<tr><td class=ar>{html.escape(first['text'])}"
                f"<div class=ph>{html.escape(first['phones'])}</div></td>"
                f"<td>{players}</td><td>{first['duration_s']}s<br>"
                f"<span class=ph>x{first['rate_ratio']}</span></td><td>{check}</td></tr>"
            )
        parts.append("</table>")

    gt = [r for r in rows if r["category"] == "ground_truth"]
    if gt:
        parts.append("<h2>ground_truth</h2><p class=desc>Real tokens decoded through "
                     "NeuCodec are the ceiling: any gap between 'truth' and 'ft' is the "
                     "model, not the codec.</p>")
        for item in items:
            if item.get("truth_wav"):
                parts.append(f"<p class=ar>{html.escape(item['text'])}</p>"
                             f"<audio controls src='wav/{item['truth_wav']}'></audio>")
    (out / "listen.html").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())