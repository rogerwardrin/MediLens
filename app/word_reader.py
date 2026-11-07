"""Single-word medicine-name reader.

Loads the EasyOCR-trainer CRNN checkpoint, reads a cropped word image two ways
(greedy decode, and CTC-likelihood scoring against a known-name vocabulary), and
flags the result when the two disagree.

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

HERE = Path(__file__).resolve().parent
TRAINER = Path(os.environ.get("MEDILENS_TRAINER", r"D:\Projects\MediLens\EasyOCR\trainer"))
CKPT = Path(os.environ.get(
    "MEDILENS_CKPT",
    str(TRAINER / "saved_models" / "medical_prescription_aug" / "best_accuracy.pth")))
CONFIG = Path(os.environ.get(
    "MEDILENS_CONFIG", str(TRAINER / "config_files" / "medical_prescription.yaml")))
VOCAB = Path(os.environ.get("MEDILENS_VOCAB", str(HERE / "vocab.txt")))

DEFAULT_THRESHOLD = 0.3   # flag when normalized edit distance(raw, vocab pick) exceeds this


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass
class Reading:
    raw: str                 # plain greedy OCR output
    match: str               # best vocabulary name
    disagreement: float      # edit_distance(raw, match) / len(match); 1.0 if raw is empty
    flagged: bool            # True -> do not trust the vocabulary match
    candidates: list = field(default_factory=list)   # top vocabulary names, best first


class WordReader:
    def __init__(self, device=None, threshold=DEFAULT_THRESHOLD):
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
        self.v_text, self.v_len = self.converter.encode(self.vocab)

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
        V = len(self.vocab)
        results = []
        for i in range(0, len(images), 32):
            chunk = images[i:i + 32]
            x = torch.stack([self._prep(im) for im in chunk]).to(self.device)
            out = self.model(x, torch.zeros(len(x), 1, dtype=torch.long, device=self.device))

            idx = out.argmax(2).view(-1).cpu()
            sizes = torch.IntTensor([out.size(1)] * len(x))
            raws = self.converter.decode_greedy(idx, sizes)

            logp = out.log_softmax(2).float().cpu()
            T, C = logp.size(1), logp.size(2)
            in_len = torch.full((V,), T, dtype=torch.int32)
            for b, raw in enumerate(raws):
                lp = logp[b][:, None, :].expand(T, V, C).contiguous()
                nll = F.ctc_loss(lp, self.v_text, in_len, self.v_len, blank=0,
                                 reduction="none", zero_infinity=False)
                nll = torch.nan_to_num(nll, nan=1e9, posinf=1e9)   # impossible names rank last
                order = torch.argsort(nll)[:top_k].tolist()
                cands = [self.vocab[k] for k in order]
                match = cands[0]
                dis = 1.0 if not raw else edit_distance(raw, match) / max(len(match), 1)
                results.append(Reading(raw, match, dis, dis > thr, cands))
        return results


if __name__ == "__main__":
    # quick test:  python word_reader.py crop1.png crop2.png
    reader = WordReader()
    paths = sys.argv[1:]
    for p, r in zip(paths, reader.read([Image.open(p) for p in paths])):
        print(f"{p}: raw={r.raw!r} match={r.match!r} disagree={r.disagreement:.2f} "
              f"{'FLAGGED' if r.flagged else 'ok'} candidates={r.candidates}")
