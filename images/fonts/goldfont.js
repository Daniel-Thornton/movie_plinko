// Bitmap renderer for the Aztec / Tower gold font packs (built by tools/build_font_pack.py).
//
//   const aztec = await GoldFont.load('images/fonts/aztec');
//   aztec.draw(ctx, 'JACKPOT', x, y, { size: 48, align: 'center' });   // y = top of line
//   el.appendChild(aztec.canvas('MOVIE NIGHT', { size: 64 }));
//   aztec.draw(ctx, ':jaguar: WIN :heart:', x, y);   // symbols by name (see meta.names)
(function (global) {
  'use strict';

  class GoldFont {
    constructor(meta, atlas) {
      this.meta = meta;
      this.atlas = atlas;
    }

    static async load(dir) {
      const slug = dir.replace(/\/+$/, '').split('/').pop();
      const meta = await (await fetch(`${dir}/${slug}.json`)).json();
      const atlas = new Image();
      atlas.src = `${dir}/${meta.atlas}`;
      await atlas.decode();
      return new GoldFont(meta, atlas);
    }

    glyph(ch) {
      const chars = this.meta.chars;
      return chars[ch] || (this.meta.caseInsensitive && chars[ch.toUpperCase()]) || null;
    }

    // Replace :name: shortcodes with their symbol characters.
    expand(text) {
      return text.replace(/:([a-z_]+):/g, (m, name) => this.meta.names[name] || m);
    }

    // Width in px at the given size (size = cap height; defaults to native).
    measure(text, { size = this.meta.capHeight, tracking = 0 } = {}) {
      const scale = size / this.meta.capHeight;
      let w = 0;
      for (const ch of this.expand(text)) {
        const g = this.glyph(ch);
        w += (g ? g.xadvance : this.meta.space) + tracking;
      }
      return Math.max(0, w - tracking) * scale;
    }

    draw(ctx, text, x, y, { size = this.meta.capHeight, tracking = 0, align = 'left' } = {}) {
      const scale = size / this.meta.capHeight;
      if (align !== 'left') x -= this.measure(text, { size, tracking }) * (align === 'center' ? 0.5 : 1);
      for (const ch of this.expand(text)) {
        const g = this.glyph(ch);
        if (g) {
          ctx.drawImage(this.atlas, g.x, g.y, g.w, g.h,
            x + g.xoffset * scale, y + g.yoffset * scale, g.w * scale, g.h * scale);
        }
        x += ((g ? g.xadvance : this.meta.space) + tracking) * scale;
      }
    }

    // Standalone canvas sized to the text (crisp on HiDPI screens).
    canvas(text, { size = this.meta.capHeight, tracking = 0, dpr = global.devicePixelRatio || 1 } = {}) {
      const scale = size / this.meta.capHeight;
      const w = Math.ceil(this.measure(text, { size, tracking }) + 24 * scale);
      const h = Math.ceil(this.meta.lineHeight * scale);
      const c = document.createElement('canvas');
      c.width = w * dpr;
      c.height = h * dpr;
      c.style.width = `${w}px`;
      c.style.height = `${h}px`;
      c.setAttribute('role', 'img');
      c.setAttribute('aria-label', text.replace(/:([a-z_]+):/g, '$1'));
      const ctx = c.getContext('2d');
      ctx.scale(dpr, dpr);
      this.draw(ctx, text, 4 * scale, 0, { size, tracking });
      return c;
    }
  }

  global.GoldFont = GoldFont;
})(window);
