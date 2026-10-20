#!/usr/bin/env python3
"""
Conversor audio (mp3, wav, ogg...) <-> WIIADPCM (DSP-ADPCM de Nintendo, estéreo,
bloques de 64 KiB intercalados), el formato de tus bombshock2 / Borne_Free2.

Requisitos: python3, numpy y ffmpeg en el PATH.

  Codificar:   python3 wiiadpcm.py encode cancion.mp3 nuevo.adpcm [--ref original.adpcm]
  Decodificar: python3 wiiadpcm.py decode original.adpcm salida.wav

Estructura del archivo (deducida de tus dumps):
  - Bloques de 0x10000 bytes alternando canal 0, canal 1, canal 0, canal 1...
  - Los dos primeros bloques empiezan con una cabecera de 0x70 bytes por canal:
      0x00  "WIIADPCM"
      0x08  u32  (canal 0: 0x00010010, canal 1: 0)   <- significado desconocido, se copia
      0x0C  u32  0x10000 (tamaño de bloque)
      0x10  u32  nº de samples        0x14 u32 nº de nibbles
      0x18  u32  sample rate (32000)  0x1C u16 loop flag, u16 formato (0, 0)
      0x20  u32  inicio (2)   0x24 u32 fin (nibbles-1)   0x28 u32 actual (2)
      0x2C  16 x s16  coeficientes DSP-ADPCM (8 pares)
      0x4C  gain(0), ps (1er byte de datos), hist1, hist2 ... resto a 0 hasta 0x70
  - Datos: frames de 8 bytes = 14 samples (cabecera predictor<<4|escala + 7 bytes de nibbles).
  - Todo el archivo se rellena con ceros hasta completar bloques de 64 KiB.
"""
import argparse
import struct
import subprocess
import sys
import wave
from concurrent.futures import ProcessPoolExecutor

import numpy as np

BLOCK = 0x10000
HDR = 0x70
SPF = 14  # samples por frame


# ----------------------------------------------------------------- utilidades
def nibble_addr(sample):
    return (sample // SPF) * 16 + sample % SPF + 2


def total_nibbles(n):
    r = n % SPF
    return (n // SPF) * 16 + (0 if r == 0 else r + 2)


def load_audio(path, rate):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "2",
           "-ar", str(rate), "-f", "s16le", "-"]
    raw = subprocess.run(cmd, check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(raw, dtype="<i2").reshape(-1, 2)


def _frames(x):
    n = len(x)
    F = -(-n // SPF)
    xe = np.zeros(2 + F * SPF)
    xe[2:2 + n] = x
    idx = np.arange(F)[:, None] * SPF + np.arange(SPF)[None, :]
    return xe, idx, F


# ------------------------------------------------- cálculo de los 8 coeficientes
def find_coefs(x, K=8):
    """Predictores de orden 2 por mínimos cuadrados + agrupación LBG (K pares)."""
    xe, idx, F = _frames(x.astype(np.float64))
    cur, p1, p2 = xe[idx + 2], xe[idx + 1], xe[idx]
    R11 = (p1 * p1).sum(1); R12 = (p1 * p2).sum(1); R22 = (p2 * p2).sum(1)
    r1 = (cur * p1).sum(1); r2 = (cur * p2).sum(1); R00 = (cur * cur).sum(1)
    del cur, p1, p2

    def solve(sel):
        a = R11[sel].sum() + 1.0
        b = R12[sel].sum()
        d = R22[sel].sum() + 1.0
        e1, e2 = r1[sel].sum(), r2[sel].sum()
        det = a * d - b * b
        return np.array([(e1 * d - e2 * b) / det, (a * e2 - b * e1) / det])

    def err(C):
        c1, c2 = C[:, 0][None, :], C[:, 1][None, :]
        return (R00[:, None] - 2 * (c1 * r1[:, None] + c2 * r2[:, None])
                + c1 * c1 * R11[:, None] + 2 * c1 * c2 * R12[:, None]
                + c2 * c2 * R22[:, None])

    C = np.array([solve(slice(None))])
    while len(C) < K:
        C = np.vstack([C + [0.02, 0.01], C - [0.02, 0.01]])
        for _ in range(12):
            a = err(C).argmin(1)
            for k in range(len(C)):
                m = a == k
                if m.any():
                    C[k] = solve(m)
    Q = np.clip(np.round(C * 2048), -32768, 32767).astype(np.int64)
    return Q


# ------------------------------------------------------------- codificación
def encode_channel(x):
    """x: int16 de un canal -> (coefs (8x2), bytes de frames)"""
    n = len(x)
    C = find_coefs(x)
    xe, idx, F = _frames(x.astype(np.float64))
    cur, p1, p2 = xe[idx + 2], xe[idx + 1], xe[idx]

    # elección en bucle abierto: escala necesaria y error de cada predictor
    scale = np.zeros((F, 8))
    sq = np.zeros((F, 8))
    for k in range(8):
        e = cur - (C[k, 0] * p1 + C[k, 1] * p2) / 2048.0
        m = np.abs(e).max(1)
        scale[:, k] = np.clip(np.ceil(np.log2(np.maximum(m, 1e-9) / 7.5)), 0, 12)
        sq[:, k] = (e * e).sum(1)
    order = np.argsort(scale * 1e12 + sq, axis=1)[:, :2].tolist()  # 2 mejores
    scale = scale.astype(int).tolist()
    del cur, p1, p2, xe, idx

    xs = np.concatenate([x.astype(np.int64),
                         np.zeros(F * SPF - n, np.int64)]).tolist()
    Cl = [(int(a), int(b)) for a, b in C]
    out = bytearray(F * 8)
    h1 = h2 = 0
    for f in range(F):
        base = f * SPF
        best = None
        for k in order[f]:
            c1, c2 = Cl[k]
            sc = scale[f][k]
            while True:
                a1, a2 = h1, h2
                nibs = []
                e_tot = 0
                clip = False
                sh = sc + 11
                half = 1 << (sh - 1)
                for i in range(base, base + SPF):
                    xv = xs[i]
                    pred = c1 * a1 + c2 * a2
                    nb = ((xv << 11) - pred + half) >> sh
                    if nb > 7:
                        nb = 7; clip = True
                    elif nb < -8:
                        nb = -8; clip = True
                    s = (pred + ((nb << sc) << 11) + 1024) >> 11
                    if s > 32767:
                        s = 32767
                    elif s < -32768:
                        s = -32768
                    d = xv - s
                    e_tot += d * d
                    nibs.append(nb & 15)
                    a2 = a1
                    a1 = s
                if clip and sc < 12:
                    sc += 1
                    continue
                break
            if best is None or e_tot < best[0]:
                best = (e_tot, k, sc, nibs, a1, a2)
        _, k, sc, nibs, h1, h2 = best
        o = f * 8
        out[o] = (k << 4) | sc
        for j in range(7):
            out[o + 1 + j] = (nibs[2 * j] << 4) | nibs[2 * j + 1]
    return C, bytes(out)


def make_header(ch, n, rate, coefs, first_byte, prefix):
    nib = total_nibbles(n)
    h = b"WIIADPCM" + prefix
    h += struct.pack(">IIIHHIII", n, nib, rate, 0, 0, 2, nibble_addr(n - 1), 2)
    h += struct.pack(">16h", *[int(v) for v in np.asarray(coefs).reshape(-1)])
    h += struct.pack(">HHhh", 0, first_byte, 0, 0)
    return h.ljust(HDR, b"\0")


def encode(args):
    pcm = load_audio(args.input, args.rate)
    n = len(pcm)
    print(f"Audio: {n} samples/canal a {args.rate} Hz = {n / args.rate:.1f} s")

    prefixes = [struct.pack(">II", 0x10010, BLOCK), struct.pack(">II", 0, BLOCK)]
    if args.ref:
        with open(args.ref, "rb") as f:
            r = f.read(BLOCK + 16)
        prefixes = [r[8:16], r[BLOCK + 8:BLOCK + 16]]

    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(encode_channel, [pcm[:, 0].copy(), pcm[:, 1].copy()]))

    headers = [make_header(c, n, args.rate, res[c][0], res[c][1][0], prefixes[c])
               for c in (0, 1)]
    first = BLOCK - HDR
    nblk = 1 + max(0, -(-(len(res[0][1]) - first) // BLOCK))
    out = bytearray()
    for b in range(nblk):
        for c in (0, 1):
            s = res[c][1]
            blk = headers[c] + s[:first] if b == 0 else s[first + (b - 1) * BLOCK:first + b * BLOCK]
            out += blk.ljust(BLOCK, b"\0")
    with open(args.output, "wb") as f:
        f.write(out)
    print(f"Escrito {args.output}: {len(out)} bytes ({nblk * 2} bloques de 64 KiB)")
    if args.ref:
        import os
        ref_size = os.path.getsize(args.ref)
        if len(out) > ref_size:
            print(f"AVISO: es {len(out) - ref_size} bytes MÁS GRANDE que el original "
                  f"({ref_size}). Si el juego/archivo contenedor espera ese tamaño, "
                  f"recorta el audio.")
        else:
            print(f"Tamaño original: {ref_size} bytes (el nuevo cabe).")


# ------------------------------------------------------------- decodificación
def decode_stream(data, coefs, n):
    c = list(coefs)
    out = np.zeros(n, np.int16)
    h1 = h2 = 0
    k = 0
    for f in range(-(-n // SPF)):
        b = data[f * 8:f * 8 + 8]
        p, sc = b[0] >> 4, b[0] & 15
        c1, c2 = c[2 * p], c[2 * p + 1]
        for i in range(SPF):
            if k >= n:
                break
            nib = (b[1 + i // 2] >> 4) if i % 2 == 0 else (b[1 + i // 2] & 15)
            if nib >= 8:
                nib -= 16
            s = (c1 * h1 + c2 * h2 + ((nib << sc) << 11) + 1024) >> 11
            s = -32768 if s < -32768 else 32767 if s > 32767 else s
            out[k] = s
            k += 1
            h2 = h1
            h1 = s
    return out


def decode(args):
    d = open(args.input, "rb").read()
    assert d[:8] == b"WIIADPCM", "No parece un WIIADPCM"
    n = struct.unpack(">I", d[0x10:0x14])[0]
    rate = struct.unpack(">I", d[0x18:0x1C])[0]
    chans = [bytearray(), bytearray()]
    for b in range(len(d) // BLOCK):
        chans[b % 2] += d[b * BLOCK + (HDR if b < 2 else 0):(b + 1) * BLOCK]
    pcm = []
    for c in (0, 1):
        coefs = struct.unpack(">16h", d[c * BLOCK + 0x2C:c * BLOCK + 0x4C])
        pcm.append(decode_stream(bytes(chans[c]), coefs, n))
    st = np.stack(pcm, 1)
    with wave.open(args.output, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(st.astype("<i2").tobytes())
    print(f"Escrito {args.output}: {n} samples/canal, {rate} Hz, {n / rate:.1f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode"); e.add_argument("input"); e.add_argument("output")
    e.add_argument("--rate", type=int, default=32000, help="por defecto 32000 (el de tus archivos)")
    e.add_argument("--ref", help="adpcm original: copia campos desconocidos y avisa si el tamaño se pasa")
    e.set_defaults(fn=encode)
    d_ = sub.add_parser("decode"); d_.add_argument("input"); d_.add_argument("output")
    d_.set_defaults(fn=decode)
    a = ap.parse_args()
    a.fn(a)
