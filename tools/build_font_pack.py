"""Slice the Aztec / Tower gold font specimen sheets into bitmap font packs.

Usage:  python tools/build_font_pack.py
Input:  images/ui/Aztec and Tower Gold Font Specimens.png          (A-Z, 0-9)
        images/ui/Aztec and Tower Gold Font Specimens SYMBOLS.png  (pictographs, punctuation)
Output: images/fonts/<font>/  glyphs/*.png, <font>.png (atlas), <font>.fnt (BMFont), <font>.json
"""
import json
import os
from collections import deque

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI = os.path.join(ROOT, 'images', 'ui')
LETTERS_SRC = os.path.join(UI, 'Aztec and Tower Gold Font Specimens.png')
SYMBOLS_SRC = os.path.join(UI, 'Aztec and Tower Gold Font Specimens SYMBOLS.png')
OUT = os.path.join(ROOT, 'images', 'fonts')

NOISE = 0.08                             # alpha below this is sheet noise / vignette
CORE_ALPHA = 0.98                        # strict threshold: glyph cores never touch
GROW_ALPHA = 0.01                        # outline/shadow pixels grown from the cores
GROW_MAX = 16                            # px a glyph may extend beyond its core (drop shadow)
TRACKING = 6                             # extra px between letters
PAD = 2                                  # atlas padding

# name: font slug -> (display name, letters band, symbols band) as source y ranges
FONTS = [('aztec', 'Aztec Gold', (100, 485), (95, 540)),
         ('tower', 'Tower Gold', (550, 941), (600, 1024))]

LETTER_ROWS = ['ABCDEFGHIJKLM', 'NOPQRSTUVWXYZ', '0123456789']
ALNUM = ''.join(LETTER_ROWS)

# Symbol sheet rows, left to right: (char, file/shortcode name, placement).
# Row 1 has no Unicode equivalents, so it lives in the Private Use Area.
PICTO = 'picto'
SYMBOL_ROWS = [
    [('', 'sun_face', PICTO), ('', 'pyramid', PICTO), ('', 'jaguar', PICTO),
     ('', 'eagle', PICTO), ('', 'serpent', PICTO), ('', 'thunderbird', PICTO),
     ('', 'cross', PICTO), ('', 'spiral', PICTO), ('', 'triangle', PICTO),
     ('', 'fret', PICTO)],
    [('♥', 'heart', PICTO), ('♠', 'spade', PICTO), ('♦', 'diamond', PICTO),
     ('♣', 'club', PICTO), ('★', 'star', PICTO), ('☾', 'moon', PICTO),
     ('☀', 'sun', PICTO), ('☥', 'ankh', PICTO), ('\U00013080', 'eye', PICTO),
     ('\U0001fab6', 'feather', PICTO)],
    [('!', 'exclam', 'base'), ('?', 'question', 'base'), (',', 'comma', 'comma'),
     (';', 'semicolon', 'comma'), (':', 'colon', 'base'), ('-', 'hyphen', 'math'),
     ('—', 'emdash', 'math'), ('@', 'at', 'middle'), ('&', 'ampersand', 'base'),
     ('$', 'dollar', 'middle'), ('%', 'percent', 'base')],
    [('(', 'parenleft', 'middle'), (')', 'parenright', 'middle'), ('[', 'bracketleft', 'middle'),
     (']', 'bracketright', 'middle'), ('{', 'braceleft', 'middle'), ('}', 'braceright', 'middle'),
     ('+', 'plus', 'math'), ('−', 'minus', 'math'), ('×', 'multiply', 'math'),
     ('÷', 'divide', 'math'), ('=', 'equal', 'math'), ('~', 'tilde', 'math'),
     ('#', 'numbersign', 'base'), ('*', 'asterisk', 'star')],
]
PUNCT_REF = '!?%'                        # these are scaled to the letters' cap height
# The sheet draws every punctuation mark the same height; relative size tweaks on top of that.
SIZE = {',': 0.55, "'": 0.45, ';': 0.85, '*': 0.8, '(': 1.15, ')': 1.15, '[': 1.15, ']': 1.15,
        '{': 1.15, '}': 1.15}
ALIASES = {'–': '-', '‑': '-', '✕': '×', '☆': '★',
           '♡': '♥', '♤': '♠', '♢': '♦', '♧': '♣'}

# Vertical placement of a symbol's core, as (anchor, fraction of cap height).
PLACEMENT = {
    'base': ('bottom', 0.0),     # sits on the baseline
    'comma': ('bottom', -0.2),   # tail drops below the baseline
    'quote': ('top', 1.0),       # hangs from the cap line
    'math': ('center', 0.45),    # math axis
    'middle': ('center', 0.5),   # centred on the caps
    'star': ('center', 0.62),
    PICTO: ('center', 0.5),
}


class Sheet:
    """A specimen sheet keyed off its green, with every glyph's pixels labelled."""

    def __init__(self, path):
        px = np.asarray(Image.open(path).convert('RGB')).astype(float)
        border = np.concatenate([px[:8].reshape(-1, 3), px[-8:].reshape(-1, 3),
                                 px[:, :8].reshape(-1, 3), px[:, -8:].reshape(-1, 3)])
        self.rgb, self.alpha = key(px, np.median(border, 0))
        core_lab, self.comps = components(self.alpha > CORE_ALPHA)
        # every seed (header ornaments too) competes for its own glow
        seeds = [c['id'] for c in self.comps if c['size'] >= 40]
        self.labels = grow(np.where(np.isin(core_lab, seeds), core_lab, 0), self.alpha > GROW_ALPHA)

    def cut(self, group, below=0):
        """RGBA crop of a group of components plus its core box (source coords)."""
        ids = [c['id'] for c in group]
        mask = np.isin(self.labels, ids)
        mask[:below] = False
        ys, xs = np.nonzero(mask & (self.alpha > CORE_ALPHA))
        core = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        ys, xs = np.nonzero(mask)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        a = np.where(mask[y0:y1, x0:x1], self.alpha[y0:y1, x0:x1], 0)
        rgba = np.dstack([self.rgb[y0:y1, x0:x1], a * 255]).round().astype(np.uint8)
        return Image.fromarray(rgba, 'RGBA'), (x0, y0), core


def key(px, bg):
    """Chroma-key the green: returns un-mixed RGB and alpha as float arrays."""
    span = bg[1] - max(bg[0], bg[2])
    excess = px[..., 1] - np.maximum(px[..., 0], px[..., 2])
    alpha = 1 - np.clip(excess / span, 0, 1)
    alpha = np.clip((alpha - NOISE) / (1 - NOISE), 0, 1)
    safe = np.maximum(alpha, 1e-3)[..., None]
    rgb = np.clip((px - (1 - alpha)[..., None] * bg) / safe, 0, 255)
    rgb[..., 1] = np.minimum(rgb[..., 1], np.maximum(rgb[..., 0], rgb[..., 2]))  # despill
    return rgb, alpha


def components(mask):
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32)
    comps = []
    for y in range(h):
        for x in np.nonzero(mask[y] & (lab[y] == 0))[0]:
            if lab[y, x]:
                continue
            n = len(comps) + 1
            lab[y, x] = n
            q = deque([(y, x)])
            y0 = y1 = y
            x0 = x1 = x
            size = 0
            while q:
                cy, cx = q.popleft()
                size += 1
                y0, y1, x0, x1 = min(y0, cy), max(y1, cy), min(x0, cx), max(x1, cx)
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not lab[ny, nx]:
                        lab[ny, nx] = n
                        q.append((ny, nx))
            comps.append(dict(id=n, y0=y0, y1=y1 + 1, x0=x0, x1=x1 + 1, size=size))
    return lab, comps


def grow(lab, mask):
    """Geodesic multi-source BFS: hand every soft pixel to its nearest core."""
    h, w = mask.shape
    out = lab.copy()
    dist = np.where(out > 0, 0, -1)
    q = deque(zip(*np.nonzero(out)))
    while q:
        y, x = q.popleft()
        n, d = out[y, x], dist[y, x] + 1
        if d > GROW_MAX:
            continue
        for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not out[ny, nx]:
                out[ny, nx] = n
                dist[ny, nx] = d
                q.append((ny, nx))
    return out


def split_rows(comps, n_rows):
    """Split components into text lines at the n_rows-1 widest vertical gaps."""
    comps = sorted(comps, key=lambda c: (c['y0'] + c['y1']) / 2)
    mids = [(c['y0'] + c['y1']) / 2 for c in comps]
    gaps = sorted(range(1, len(mids)), key=lambda i: mids[i] - mids[i - 1])[-(n_rows - 1):]
    cuts = [0] + sorted(gaps) + [len(comps)]
    return [comps[a:b] for a, b in zip(cuts, cuts[1:])]


def group_columns(row):
    """Merge components that overlap horizontally (the dot of '!', the bars of '=')."""
    groups = []
    for c in sorted(row, key=lambda c: c['x0']):
        if groups and c['x0'] <= max(g['x1'] for g in groups[-1]) + 6:
            groups[-1].append(c)
        else:
            groups.append([c])
    return groups


def glyph(ch, name, img, crop, core, scale=1.0):
    """A glyph record: scaled image plus its crop offset from the core box."""
    if scale != 1.0:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                         Image.LANCZOS)
    return dict(ch=ch, name=name, img=img,
                dx=round((crop[0] - core[0]) * scale), dy=round((crop[1] - core[1]) * scale),
                core_w=round((core[2] - core[0]) * scale), core_h=round((core[3] - core[1]) * scale))


def letter_glyphs(sheet, band):
    lo, hi = band
    cores = [c for c in sheet.comps if c['size'] >= 3000 and lo <= (c['y0'] + c['y1']) / 2 < hi]
    rows = split_rows(cores, len(LETTER_ROWS))
    assert [len(r) for r in rows] == [len(s) for s in LETTER_ROWS], [len(r) for r in rows]
    out = []
    for chars, row in zip(LETTER_ROWS, rows):
        row.sort(key=lambda c: c['x0'])
        baseline = int(np.median([c['y1'] for c in row]))
        for ch, c in zip(chars, row):
            g = glyph(ch, ch, *sheet.cut([c]))
            g['core_top'] = c['y0'] - baseline
            out.append(g)
    return out


def symbol_glyphs(sheet, band, cap):
    lo, hi = band
    parts = [c for c in sheet.comps if c['size'] >= 40 and lo <= (c['y0'] + c['y1']) / 2 < hi]
    rows = [group_columns(r) for r in split_rows(parts, len(SYMBOL_ROWS))]
    assert [len(r) for r in rows] == [len(s) for s in SYMBOL_ROWS], [len(r) for r in rows]

    cuts = {}
    for spec, row in zip(SYMBOL_ROWS, rows):
        for (ch, name, place), group in zip(spec, row):
            cuts[ch] = (name, place, group, sheet.cut(group))
    height = lambda ch: cuts[ch][3][2][3] - cuts[ch][3][2][1]
    punct_scale = cap / np.median([height(ch) for ch in PUNCT_REF])
    picto_scale = cap / np.median([height(ch) for ch, (_, p, _, _) in cuts.items() if p == PICTO])

    out = []
    for ch, (name, place, group, cut) in cuts.items():
        scale = picto_scale if place == PICTO else punct_scale * SIZE.get(ch, 1)
        out.append(dict(glyph(ch, name, *cut, scale=scale), place=place))
    # No period or apostrophe on the sheet: take the colon's lower dot and a raised comma.
    _, _, group, (_, _, (x0, y0, x1, y1)) = cuts[':']
    ink = (np.isin(sheet.labels, [c['id'] for c in group]) & (sheet.alpha > CORE_ALPHA))[y0:y1, x0:x1]
    third = (y1 - y0) // 3
    waist = y0 + third + int(np.argmin(ink.sum(1)[third:-third]))
    out.append(dict(glyph('.', 'period', *sheet.cut(group, below=waist), scale=punct_scale),
                    place='base'))
    out.append(dict(glyph("'", 'quotesingle', *cuts[','][3], scale=punct_scale * SIZE["'"]),
                    place='quote'))

    for g in out:
        anchor, frac = PLACEMENT[g['place']]
        y = -frac * cap
        g['core_top'] = round({'bottom': y - g['core_h'], 'top': y, 'center': y - g['core_h'] / 2}[anchor])
    return out


def pack(glyphs):
    """Simple shelf packer; returns atlas size and sets g['ax'], g['ay']."""
    width = 1024
    x = y = shelf = 0
    for g in sorted(glyphs, key=lambda g: -g['img'].height):
        w, h = g['img'].size
        if x + w + PAD > width:
            x, y, shelf = 0, y + shelf + PAD, 0
        g['ax'], g['ay'] = x + PAD, y + PAD
        x += w + PAD
        shelf = max(shelf, h)
    height = 1
    while height < y + shelf + 2 * PAD:
        height *= 2
    return width, height


def write_font(slug, name, glyphs):
    for g in glyphs:
        g['top'] = g['core_top'] + g['dy']           # crop top relative to baseline
        g['xadvance'] = g['core_w'] + TRACKING
    letters = [g for g in glyphs if g['ch'] in ALNUM]
    ascent = -min(g['top'] for g in letters)
    descent = max(g['top'] + g['img'].height for g in letters)
    line_height = ascent + descent
    cap = int(np.median([-g['top'] for g in letters if g['ch'].isalpha()]))
    space = round(cap * 0.35)

    out_dir = os.path.join(OUT, slug)
    os.makedirs(os.path.join(out_dir, 'glyphs'), exist_ok=True)
    for g in glyphs:
        g['img'].save(os.path.join(out_dir, 'glyphs', f"{g['name']}.png"), optimize=True)

    aw, ah = pack(glyphs)
    atlas = Image.new('RGBA', (aw, ah))
    for g in glyphs:
        atlas.paste(g['img'], (g['ax'], g['ay']))
    atlas.save(os.path.join(out_dir, f'{slug}.png'), optimize=True)

    glyphs.sort(key=lambda g: ord(g['ch']))
    chars = {g['ch']: {
        'x': g['ax'], 'y': g['ay'], 'w': g['img'].width, 'h': g['img'].height,
        'xoffset': g['dx'], 'yoffset': g['top'] + ascent, 'xadvance': g['xadvance'],
    } for g in glyphs}
    chars.update({a: chars[t] for a, t in ALIASES.items()})
    meta = {
        'name': name, 'atlas': f'{slug}.png', 'lineHeight': line_height, 'base': ascent,
        'capHeight': cap, 'space': space, 'caseInsensitive': True,
        'names': {g['name']: g['ch'] for g in glyphs if g['ch'] not in ALNUM},
        'chars': chars,
    }
    with open(os.path.join(out_dir, f'{slug}.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)

    lines = [
        f'info face="{name}" size={line_height} bold=0 italic=0 charset="" unicode=1 '
        f'stretchH=100 smooth=1 aa=1 padding=0,0,0,0 spacing={PAD},{PAD}',
        f'common lineHeight={line_height} base={ascent} scaleW={aw} scaleH={ah} pages=1 packed=0',
        f'page id=0 file="{slug}.png"',
        f'chars count={len(chars) + 1}',
        f'char id=32 x=0 y=0 width=0 height=0 xoffset=0 yoffset=0 xadvance={space} page=0 chnl=15',
    ]
    for ch, c in sorted(chars.items(), key=lambda kv: ord(kv[0])):
        lines.append(f"char id={ord(ch)} x={c['x']} y={c['y']} width={c['w']} height={c['h']} "
                     f"xoffset={c['xoffset']} yoffset={c['yoffset']} xadvance={c['xadvance']} page=0 chnl=15")
    with open(os.path.join(out_dir, f'{slug}.fnt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'{name}: {len(glyphs)} glyphs, atlas {aw}x{ah}, lineHeight {line_height}, cap {cap}')


def main():
    letters, symbols = Sheet(LETTERS_SRC), Sheet(SYMBOLS_SRC)
    for slug, name, letter_band, symbol_band in FONTS:
        glyphs = letter_glyphs(letters, letter_band)
        cap = np.median([-g['core_top'] for g in glyphs if g['ch'].isalpha()])
        glyphs += symbol_glyphs(symbols, symbol_band, cap)
        write_font(slug, name, glyphs)


if __name__ == '__main__':
    main()
