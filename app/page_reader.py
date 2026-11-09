"""Full-page reading: CRAFT detection -> crop -> WordReader."""
import numpy as np
import easyocr

PAD, MIN_H, MIN_W = 4, 12, 24


class PageReader:
    def __init__(self, word_reader, gpu=True):
        self.wr = word_reader
        self.det = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def boxes(self, img):
        h_list, f_list = self.det.detect(np.array(img))
        out = [[int(x0), int(y0), int(x1), int(y1)] for x0, x1, y0, y1 in h_list[0]]
        for poly in f_list[0]:
            p = np.array(poly)
            out.append([int(p[:, 0].min()), int(p[:, 1].min()), int(p[:, 0].max()), int(p[:, 1].max())])
        out = [[max(0, a), max(0, b), c, d] for a, b, c, d in out]
        out = [b for b in out if b[3] - b[1] >= MIN_H and b[2] - b[0] >= MIN_W]
        return sorted(out, key=lambda b: (b[1], b[0]))

    def read(self, img, threshold=None):
        img = img.convert("RGB")
        W, H = img.size
        boxes = self.boxes(img)
        crops = [img.crop((max(0, x0 - PAD), max(0, y0 - PAD), min(W, x1 + PAD), min(H, y1 + PAD)))
                 for x0, y0, x1, y1 in boxes]
        readings = self.wr.read(crops, threshold=threshold, gate = True) if crops else []
        return boxes, crops, readings