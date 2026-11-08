"""Medicine-name reader: reads a cropped word image and returns form + name.

Loads the EasyOCR-trainer CRNN checkpoint and reads a crop two ways:
  1. greedy decode (plain OCR text), and
  2. CTC-likelihood scoring of every known name, bare and with a written form prefix
     ("Tab ", "Cap. " ...). The best candidate gives the name AND the form.
If the two readings disagree strongly, the result is flagged for review.

Configure with environment variables, or edit the defaults below:
    MEDILENS_TRAINER  folder containing the EasyOCR trainer's model.py / utils.py
    MEDILENS_CKPT     path to best_accuracy.pth
    MEDILENS_CONFIG   path to medical_prescription.yaml
    MEDILENS_VOCAB    text file with one known medicine name per line
"""
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from prescription_parser import DEFAULT_FORMS, build_candidates, edit_distance

HERE = Path(__file__).resolve().parent
TRAINER = Path(os.environ.get("MEDILENS_TRAINER", r"D:\Projects\MediLens\EasyOCR\trainer"))
CKPT = Path(os.environ.get(
    "MEDILENS_CKPT",
    r"D:\Projects\MediLens\models\kaggle_e400_fall_best.pth"))
CONFIG = Path(os.environ.get(
    "MEDILENS_CONFIG", str(TRAINER / "config_files" / "medical_prescription.yaml")))
VOCAB = Path(os.environ.get("MEDILENS_VOCAB", str(HERE / "vocab.txt")))

DEFAULT_THRESHOLD = 2.30   # flag when margin (gap to the 2nd-best name, length-normalised NLL) is below this
CHUNK = 256               # candidates scored per CTC call (bounds memory)


@dataclass
class Reading:
    raw: str                 # plain greedy OCR output (includes any written 'Tab'/'Cap')
    name: str                # best known name
    form: str                # 'tablet' | 'capsule' | ... | None (no form written)
    surface: str             # the winning candidate text, e.g. 'Tab. Covam'
    disagreement: float      # edit_distance(raw, surface) / len(surface); 1.0 if raw is empty
    margin: float            # score gap between the best and second-best known name
    flagged: bool            # True -> do not trust the match
    candidates: list = field(default_factory=list)   # top known names, best first

    @property
    def match(self):         # kept for older code
        return self.name


class WordReader:
    def __init__(self, device=None, threshold=DEFAULT_THRESHOLD, forms=DEFAULT_FORMS):
        for p in (TRAINER / "model.py", TRAINER / "utils.py", CKPT, CONFIG, VOCAB):
            if not Path(p).exists():
                raise FileNotFoundError(f"Missing file: {p}")
        if str(TRAINER) not in sys.path:
            sys.path.insert(0, str(TRAINER))
        from model import Model                    # EasyOCR trainer modules
        from utils import CTCLabelConverter

        self.threshold = threshold
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        with open(CONFIG, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        character = cfg["number"] + cfg["symbol"] + cfg["lang_char"]
        self.converter = CTCLabelConverter(character)
        opt = SimpleNamespace(**cfg)
        opt.character = character
        opt.num_class = len(self.converter.character)
        self.H, self.W = int(cfg["imgH"]), int(cfg["imgW"])

        model = Model(opt)
        state = torch.load(CKPT, map_location="cpu")
        state = {k.replace("module.", "", 1): v for k, v in state.items()}   # saved from DataParallel
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()

        names = [ln.strip() for ln in VOCAB.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.vocab = sorted(set(names))
        allowed = set(character)
        bad = {c for n in self.vocab for c in n if c not in allowed}
        if bad:
            raise ValueError(f"Vocabulary has characters the model cannot output: {sorted(bad)}")

        # every name, bare and with each form prefix (name-major order)
        self.cands, self.n_var = build_candidates(self.vocab, forms)
        self.cand_len = np.array([len(c[0]) for c in self.cands], dtype=np.float32)
        self.chunks = []
        for s in range(0, len(self.cands), CHUNK):
            texts = [c[0] for c in self.cands[s:s + CHUNK]]
            v_text, v_len = self.converter.encode(texts)
            self.chunks.append((len(texts), v_text, v_len))

    def _prep(self, im: Image.Image) -> torch.Tensor:
        im = im.convert("L")
        w, h = im.size
        nw = min(self.W, max(1, round(self.H * w / h)))
        arr = np.asarray(im.resize((nw, self.H), Image.BICUBIC), dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        out = np.repeat(arr[:, -1:], self.W, axis=1)   # pad by repeating the last column
        out[:, :nw] = arr
        return torch.from_numpy(out)[None]

    @torch.no_grad()
    def read(self, images, threshold=None, top_k=3):
        thr = self.threshold if threshold is None else threshold
        starts = np.arange(len(self.vocab)) * self.n_var
        results = []
        for i in range(0, len(images), 16):
            x = torch.stack([self._prep(im) for im in images[i:i + 16]]).to(self.device)
            out = self.model(x, torch.zeros(len(x), 1, dtype=torch.long, device=self.device))

            # plain greedy decode (collapse repeats, drop blanks)
            raws = []
            for row in out.argmax(2).cpu().tolist():
                s, prev = [], 0
                for k in row:
                    if k != 0 and k != prev:
                        s.append(self.converter.character[k])
                    prev = k
                raws.append("".join(s))

            logp = out.log_softmax(2).float().cpu()
            T, C = logp.size(1), logp.size(2)
            for b, raw in enumerate(raws):
                nll = np.empty(len(self.cands), dtype=np.float32)
                col = 0
                for n, v_text, v_len in self.chunks:
                    lp = logp[b][:, None, :].expand(T, n, C).contiguous()
                    loss = F.ctc_loss(lp, v_text, torch.full((n,), T, dtype=torch.int32), v_len,
                                      blank=0, reduction="none", zero_infinity=False)
                    nll[col:col + n] = torch.nan_to_num(loss, nan=1e9, posinf=1e9).numpy()
                    col += n
                nll = nll / self.cand_len                     # length-normalised, as in the evaluation
                best = int(nll.argmin())
                surface, name, form = self.cands[best]
                per_name = np.minimum.reduceat(nll, starts)   # best score per name, any form
                order = np.argsort(per_name)
                top = [self.vocab[k] for k in order[:top_k]]
                margin = float(per_name[order[1]] - per_name[order[0]])
                dis = 1.0 if not raw else edit_distance(raw, surface) / max(len(surface), 1)
                results.append(Reading(raw=raw, name=name, form=form, surface=surface,
                                        disagreement=dis, margin=margin,
                                        flagged=margin < thr, candidates=top))
        return results


if __name__ == "__main__":
    # quick test:  python word_reader.py crop1.png crop2.png
    reader = WordReader()
    paths = sys.argv[1:]
    for p, r in zip(paths, reader.read([Image.open(p) for p in paths])):
        print(f"{p}: raw={r.raw!r} -> form={r.form} name={r.name!r} disagree={r.disagreement:.2f} "
              f"{'FLAGGED' if r.flagged else 'ok'} candidates={r.candidates}")
