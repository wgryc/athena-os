# STYLE.md

This document tracks our styling decisions to ensure we are consistent across repositories, etc.

Refer to [style.html](style.html) for more thorough examples, HTML code, etc.

---

## Color Palette, "Deep Sea Ink"

A color scheme pairing cool, crisp blue-gray surfaces with warm organic accent colors. Designed for dashboards, prose, and code — comfortable for extended screen use in both light and dark modes.

The backgrounds and text are clean and cool, giving the interface a modern, professional feel. The accents draw from warm natural tones — ink reds, burnt orange, honey-gold, emerald, teal — so data and highlights feel grounded and alive rather than sterile. The contrast between cool structure and warm detail creates strong visual hierarchy without fatigue.

All accent colors shift to lighter variants in dark mode to maintain readability and WCAG AA contrast ratios.

---

### Light Mode

| Role       | Hex       | Notes                        |
|------------|-----------|------------------------------|
| Background | `#F6F8FC` | Cool off-white               |
| Surface    | `#FFFFFF` | Pure white cards             |
| Surface 2  | `#F1F4FB` | Subtle blue-tinted secondary |
| Text       | `#0E1A2B` | Near-black, cool             |
| Muted text | `#4B5B73` | Cool gray for secondary text |
| Border     | `rgba(14, 26, 43, 0.12)` | Transparent dark   |

#### Light Mode Accents

| Color   | Hex       | Semantic Use            |
|---------|-----------|-------------------------|
| Red     | `#AF3029` | Errors, invalid imports |
| Orange  | `#BC5215` | Warnings, functions     |
| Yellow  | `#C08B2C` | Constants, caution      |
| Green   | `#3B855C` | Success, keywords       |
| Cyan    | `#2B7E78` | Links, strings          |
| Blue    | `#205EA6` | Primary, variables      |
| Purple  | `#5E409D` | Numbers, special        |
| Magenta | `#A02F6F` | Language features       |

---

### Dark Mode

| Role       | Hex       | Notes                         |
|------------|-----------|-------------------------------|
| Background | `#0B1020` | Deep navy-black               |
| Surface    | `#121A33` | Dark blue card surface        |
| Surface 2  | `#182245` | Elevated/hover surface        |
| Text       | `#E8EEFC` | Soft blue-white               |
| Muted text | `#B8C6EE` | Lavender-gray secondary       |
| Border     | `rgba(255, 255, 255, 0.10)` | Transparent light |

#### Dark Mode Accents

| Color   | Hex       | Semantic Use            |
|---------|-----------|-------------------------|
| Red     | `#D14D41` | Errors, invalid imports |
| Orange  | `#DA702C` | Warnings, functions     |
| Yellow  | `#E2B53E` | Constants, caution      |
| Green   | `#5CB97A` | Success, keywords       |
| Cyan    | `#42AEA0` | Links, strings          |
| Blue    | `#4385BE` | Primary, variables      |
| Purple  | `#8B7EC8` | Numbers, special        |
| Magenta | `#CE5D97` | Language features       |

---

### CSS Variables — Colors

```css
/* Deep Sea Ink — Light Mode */
:root {
  --bg: #F6F8FC;
  --surface: #FFFFFF;
  --surface-2: #F1F4FB;
  --text: #0E1A2B;
  --text-muted: #4B5B73;
  --border: rgba(14, 26, 43, 0.12);

  --red: #AF3029;
  --orange: #BC5215;
  --yellow: #C08B2C;
  --green: #3B855C;
  --cyan: #2B7E78;
  --blue: #205EA6;
  --purple: #5E409D;
  --magenta: #A02F6F;
}

/* Deep Sea Ink — Dark Mode */
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0B1020;
    --surface: #121A33;
    --surface-2: #182245;
    --text: #E8EEFC;
    --text-muted: #B8C6EE;
    --border: rgba(255, 255, 255, 0.10);

    --red: #D14D41;
    --orange: #DA702C;
    --yellow: #E2B53E;
    --green: #5CB97A;
    --cyan: #42AEA0;
    --blue: #4385BE;
    --purple: #8B7EC8;
    --magenta: #CE5D97;
  }
}
```

---

### Tailwind Config — Colors

```js
colors: {
  bg: 'var(--bg)',
  surface: 'var(--surface)',
  'surface-2': 'var(--surface-2)',
  text: 'var(--text)',
  muted: 'var(--text-muted)',
  border: 'var(--border)',
  red: 'var(--red)',
  orange: 'var(--orange)',
  yellow: 'var(--yellow)',
  green: 'var(--green)',
  cyan: 'var(--cyan)',
  blue: 'var(--blue)',
  purple: 'var(--purple)',
  magenta: 'var(--magenta)',
}
```

---

## Typography

We use an all-sans system. A single sans-serif family (DM Sans) handles every role — headings, body, UI, labels. Visual hierarchy comes from weight and size, not font contrast. JetBrains Mono is used only for code and numerical data.

Both fonts are open source (SIL Open Font License) and available on Google Fonts. DM Sans is a variable font with an optical size axis — it subtly adjusts weight distribution at different sizes for optimal readability.

| Role | Font           | Use                                             |
|------|----------------|-------------------------------------------------|
| Sans | DM Sans        | Everything: headings, body, nav, labels, buttons, forms |
| Mono | JetBrains Mono | Code blocks, data tables, KPI values, terminal  |

---

### Loading

Google Fonts CDN:

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800;1,9..40,400&family=JetBrains+Mono:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
```

### CSS Variables — Typography

```css
:root {
  --font-sans: 'DM Sans', sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}
```

### Tailwind Config — Typography

```js
fontFamily: {
  sans: ['DM Sans', 'sans-serif'],
  mono: ['JetBrains Mono', 'monospace'],
}
```

### Fallback Stack

```css
--font-sans: 'DM Sans', 'Segoe UI', system-ui, sans-serif;
--font-mono: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
```

---

### Weight Scale

DM Sans supports weights 400–800. Use this subset:

| Weight | Name       | Use                                           |
|--------|------------|-----------------------------------------------|
| 400    | Regular    | Body paragraphs, descriptions, helper text    |
| 500    | Medium     | Nav items, form labels, table headers         |
| 600    | SemiBold   | Subheadings, emphasis, buttons, active states |
| 700    | Bold       | Section headings, card titles                 |
| 800    | ExtraBold  | Hero headings, page titles                    |

---

### Heading Hierarchy

Headings use relaxed negative tracking — just enough to feel intentional without crowding. DM Sans has comfortable default spacing, so only light tightening is needed.

| Level | Weight | Size   | Letter Spacing | Line Height | Margin Bottom | Use                        |
|-------|--------|--------|----------------|-------------|---------------|----------------------------|
| H1    | 800    | 3rem   | -0.015em       | 1.1         | 8px           | Page hero, landing headers |
| H2    | 700    | 1.75rem| -0.01em        | 1.2         | 8px           | Section titles             |
| H3    | 700    | 1.35rem| -0.01em        | 1.3         | 8px           | Card titles, subsections   |
| H4    | 600    | 1.05rem| 0              | 1.4         | 6px           | Sidebar titles, minor headings |

**Subtitles** appear directly below a heading. They use:
- Weight: **600**
- Size: roughly half to two-thirds the heading size (e.g. 1.5rem under an H1)
- Color: `--text-muted`
- Letter spacing: **-0.005em**
- Margin top: **0** (tight coupling with heading)

---

### Body Text

| Context           | Weight | Size    | Line Height | Color          | Max Width |
|-------------------|--------|---------|-------------|----------------|-----------|
| Primary body      | 400    | 1rem    | 1.7         | `--text`       | 640px     |
| Secondary / small | 400    | 0.88rem | 1.65        | `--text-muted` | 640px     |

Body paragraphs should have **12–14px** spacing between them (`margin-bottom`). Keep line length under 640px (~65–75 characters) for comfortable reading.

Inline emphasis uses **weight 600** and the accent color `--cyan` instead of italics. This provides stronger visual contrast in an all-sans system.

---

### Navigation

Nav items sit in a horizontal row with consistent spacing:

| Element      | Weight | Size    | Color          | Spacing Between |
|-------------|--------|---------|----------------|-----------------|
| Brand name  | 700    | 1rem    | `--text`       | —               |
| Nav items   | 500    | 1rem    | `--text-muted` | 24px gap        |
| Active item | 600    | 1rem    | `--blue`       | —               |

---

### Buttons & Actions

| Variant   | Weight | Size    | Padding       | Border Radius | Background  | Text Color |
|-----------|--------|---------|---------------|---------------|-------------|------------|
| Primary   | 600    | 0.88rem | 10px 20px     | 8px           | `--blue`    | `#FFFFFF`  |
| Secondary | 600    | 0.88rem | 10px 20px     | 8px           | `--surface-2` | `--text` |
| Success   | 600    | 0.88rem | 10px 20px     | 8px           | `--green`   | `#FFFFFF`  |
| Danger    | 600    | 0.88rem | 10px 20px     | 8px           | `--red`     | `#FFFFFF`  |

Letter spacing on buttons: **-0.01em** (slight tightening reads more polished).

---

### Form Elements

| Element     | Weight | Size    | Color        | Notes                                       |
|-------------|--------|---------|--------------|----------------------------------------------|
| Field label | 600    | 0.8rem  | `--text-muted` | Displayed above input, 4px margin below    |
| Input text  | 400    | 0.95rem | `--text`     | Padding: 10px 14px, border-radius: 8px      |
| Placeholder | 400    | 0.95rem | `--text-muted` | Same size as input text, lighter color     |

Input fields use `--bg` background with `--border` border. On focus, border becomes `--blue`.

---

### KPI / Metric Cards

| Element     | Font | Weight | Size    | Style                                     |
|-------------|------|--------|---------|-------------------------------------------|
| Label       | Sans | 600    | 0.7rem  | Uppercase, `letter-spacing: 0.08em`, `--text-muted` |
| Value       | Mono | 700    | 1.5rem  | `letter-spacing: -0.02em`, `--text`       |
| Delta       | Mono | 600    | 0.75rem | Green for positive, yellow for caution, red for negative |

Cards use `--bg` background, `--border` border, 10px border-radius, 14px 16px padding.

---

### Section Labels

Small uppercase labels used to divide content areas (e.g. "Dashboard KPIs", "Code", "Navigation"):

- Font: **JetBrains Mono**
- Weight: **600**
- Size: **0.65rem**
- Style: uppercase, `letter-spacing: 0.1em`
- Color: `--blue`
- Margin: **24px top**, **12px bottom** (0 top if first element)

---

### Data Tables

| Element     | Font | Weight | Size    | Notes                              |
|-------------|------|--------|---------|------------------------------------|
| Header      | Sans | 600    | 0.75rem | Uppercase, `letter-spacing: 0.06em`, `--text-muted` |
| Cell text   | Sans | 400    | 0.88rem | `--text`                           |
| Cell data   | Mono | 400    | 0.82rem | For numbers, IDs, codes, methods   |

Table rows separated by `--border` bottom borders. Header row has a 2px bottom border. No vertical borders.

---

### Code Blocks

- Font: **JetBrains Mono**
- Weight: **400**
- Size: **0.82rem**
- Line height: **1.7**
- Background: `--surface-2`
- Border radius: **10px**
- Padding: **18px 22px**

Syntax colors follow the accent color mappings: green for keywords, cyan for strings, orange for functions, purple for numbers, blue for variables, `--text-muted` italic for comments.

---

### Spacing Reference

| Context                          | Value    |
|----------------------------------|----------|
| Card padding                     | 16–22px  |
| Card border-radius               | 10–16px  |
| Gap between KPI cards            | 12px     |
| Gap between sections             | 24px     |
| Gap between nav items            | 24px     |
| Body paragraph margin-bottom     | 12–14px  |
| Heading to subtitle gap          | 0–4px    |
| Heading margin-bottom            | 8px      |
| Section label to content gap     | 12px     |
| Button row gap                   | 10px     |
| Form field row gap               | 14px     |
| Label to input gap               | 4px      |