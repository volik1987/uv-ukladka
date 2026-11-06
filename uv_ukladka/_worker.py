"""Процесс-помощник «UV Укладки» (свой Python-процесс — свой GIL, все ядра заняты).
  _worker.py build <файл>      — построить маски островов
  _worker.py <файл> <ширина> <порядок> — одна проба раскладки в полосу
Ответ — pickle в stdout."""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import packer  # noqa: E402


def main():
    if sys.argv[1] == "build":
        with open(sys.argv[2], "rb") as f:
            items, k_base, g, angles = pickle.load(f)
        out = [[v.plain() for v in vs] for vs in packer.build_chunk(items, k_base, g, angles)]
    else:
        path, W, oi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
        with open(path, "rb") as f:
            plain, orders = pickle.load(f)
        variants = packer.from_plain(plain)
        out = packer.pack_once([variants[i] for i in orders[oi]], W, Hc=int(W * 2.5) + 8)
    sys.stdout.buffer.write(pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL))


if __name__ == "__main__":
    main()
