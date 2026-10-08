# Harvesting GLEAM 3 feed intake from the public dashboard

The livestock feed share needs dry-matter intake by feed category, per country. FAO publishes it
only through the GLEAM 3 dashboard, which is a Shiny app with no bulk download. This is how to
get it out, and what the result can and cannot be used for.

## What the dashboard serves

All **eight** feed categories, but not all of them on every species' table, which is the trap:
look at cattle alone and you will conclude two are missing.

| species | categories on its table |
|---|---|
| buffalo, cattle, goats, sheep | By-products, Crop residues, Fodder crop, Grains, Grass and leaves, Oil seed cakes |
| chickens | By-products, Grains, Grass and leaves, Oil seed cakes, **Other edible**, **Other non-edible** |
| pigs | By-products, Crop residues, Grains, Grass and leaves, Oil seed cakes, **Other edible**, **Other non-edible** |

Ruminants get fodder crop and no "other" categories; monogastrics get the two "other" categories
and no fodder crop. Their union is the full eight, so a harvest across every species yields a
real feed share rather than an upper bound on one.

This matters when parsing: the columns shift position by species, so the values cannot be read
by column index against a single header. The harvest below records the species on every row, and
the loader maps each row against that species' own layout.

The compiled extract (`gleam3_dmi.xlsx`) is still worth having, to validate this against.

## Why it cannot be scripted headlessly

The app is Shiny over a websocket, and it will not hand out a session to a plain HTTP client
(`RuntimeError: no session id`). It also suspends rendering for any client that reports itself
hidden, so a background or offscreen browser tab returns an empty table forever. The window has
to be genuinely visible while the harvest runs.

## Steps

1. Open <https://foodandagricultureorganization.shinyapps.io/GLEAMV3_Public/> in Chrome, and
   leave the window visible and frontmost for the whole run. Do not minimise it or switch to a
   different desktop.
2. Click **Input Data** in the left sidebar. Four sections appear, the last being **Dry Matter
   Intake**. They stay empty until a country is selected: the World-level table is not populated.
3. Pick any country in the **Area** selector and confirm the Dry Matter Intake table fills. The
   species radios above it filter the table to one species at a time, so the harvest walks
   country by country and species by species.
4. Open the browser console and paste the script below. It records the species on every row,
   which is what lets the loader map each row against that species' own column layout.
5. Watch progress with `__gleam.done + " / " + __gleam.total`. Expect about twenty minutes for
   231 countries.
6. When `__gleam.finished` is true, run the export line. It prints one pipe-separated row per
   country with all eight categories; save that as
   `base_data/global_invest/livestock_provision/gleam3_dmi_dashboard.psv`.

Downloads and network calls out of the page are both blocked by the app's content-security
policy, so the export prints to the console rather than saving a file.

## The script

```javascript
(() => {
  const el = document.querySelector('#selectCountry'), opts = el.selectize.options;
  const agg = /^(WORLD|CONTINE|FAOREG|GLEAMR|SDGsRE|EU27|LDC|LLDC|SIDS|OECD|G20)/i;
  const countries = Object.keys(opts).filter(k => !agg.test(k))
      .map(k => ({code: k, label: (opts[k].label || opts[k].text || '')}));
  window.__gleam = {rows: [], heads: null, done: 0, total: countries.length, empty: []};
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const host = () => document.querySelector('#inputDryMatterIntakeSummary');
  const read = () => {
    const h = host(); if (!h) return null;
    const b = h.querySelector('.dataTables_scrollBody tbody') || h.querySelector('tbody');
    if (!b) return null;
    return {
      heads: [...h.querySelectorAll('thead th')].map(x => x.innerText.trim()).filter(Boolean),
      rows: [...b.querySelectorAll('tr')].map(r => [...r.cells].map(c => c.innerText.trim()))
              .filter(r => r.length > 4),
    };
  };
  (async () => {
    for (const c of countries) {
      el.selectize.setValue(c.code, false);
      Shiny.setInputValue('selectCountry', c.code, {priority: 'event'});
      await sleep(900);
      let any = false;
      for (const radio of [...document.querySelectorAll('input[name=inData_selSpecies]')]) {
        radio.click();
        let got = null;
        for (let i = 0; i < 25; i++) {
          await sleep(350);
          const t = read();
          if (t && t.rows.length && t.rows[0][1] === c.label && t.rows[0][2] === radio.value) {
            got = t; break;
          }
        }
        if (!got) continue;
        window.__gleam.heads = got.heads;
        for (const r of got.rows) window.__gleam.rows.push([c.code, radio.value, ...r]);
        any = true;
      }
      if (!any) window.__gleam.empty.push(c.code);
      window.__gleam.done++;
    }
    window.__gleam.finished = true;
  })();
  return 'harvest started';
})()
```

## The export line

Each species' row is mapped against its own layout, then summed per country.

```javascript
(() => {
  const LAYOUT = {
    Buffalo: ['By-products','Crop residues','Fodder crop','Grains','Grass and leaves','Oil seed cakes'],
    Cattle:  ['By-products','Crop residues','Fodder crop','Grains','Grass and leaves','Oil seed cakes'],
    Goats:   ['By-products','Crop residues','Fodder crop','Grains','Grass and leaves','Oil seed cakes'],
    Sheep:   ['By-products','Crop residues','Fodder crop','Grains','Grass and leaves','Oil seed cakes'],
    Chickens:['By-products','Grains','Grass and leaves','Oil seed cakes','Other edible','Other non-edible'],
    Pigs:    ['By-products','Crop residues','Grains','Grass and leaves','Oil seed cakes','Other edible','Other non-edible'],
  };
  const ALL = ['By-products','Crop residues','Fodder crop','Grains','Grass and leaves',
               'Oil seed cakes','Other edible','Other non-edible'];
  const per = {}, mismatch = {};
  for (const r of window.__gleam.rows) {
    const code = r[0], species = r[1], cells = r.slice(6), layout = LAYOUT[species];
    if (!layout || cells.length !== layout.length) {
      mismatch[species + ':' + cells.length] = (mismatch[species + ':' + cells.length] || 0) + 1;
      continue;                       // never guess a layout: a wrong one sums the wrong columns
    }
    if (!per[code]) per[code] = Object.fromEntries(ALL.map(c => [c, null]));
    layout.forEach((cat, i) => {
      const s = String(cells[i]).replace(/,/g, '').trim();
      if (s !== '') { const v = Number(s); if (!Number.isNaN(v)) per[code][cat] = (per[code][cat] || 0) + v; }
    });
  }
  console.log('unmapped rows (must be empty):', mismatch);
  const lines = ['country_code|' + ALL.join('|')];
  for (const code of Object.keys(per).sort())
    lines.push(code + '|' + ALL.map(c => per[code][c] === null ? '' : per[code][c]).join('|'));
  console.log(lines.join('\n'));
  return Object.keys(per).length + ' countries';
})()
```

## Checking what came back

The export prints `unmapped rows`, which **must be empty**: a row whose cell count does not match
its species layout has been skipped rather than summed into the wrong category. Countries in
`__gleam.empty` returned no table at all, which is normal for small territories GLEAM does not
model (about two dozen of the 231).
