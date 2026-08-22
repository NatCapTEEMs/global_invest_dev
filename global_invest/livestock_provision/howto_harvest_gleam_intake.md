# Harvesting GLEAM 3 feed intake from the public dashboard

The livestock feed share needs dry-matter intake by feed category, per country. FAO publishes it
only through the GLEAM 3 dashboard, which is a Shiny app with no bulk download. This is how to
get it out, and what the result can and cannot be used for.

## Read this before running it

The dashboard serves **six** feed categories:

    By-products, Crop residues, Fodder crop, Grains, Grass and leaves, Oil seed cakes

The method needs **eight**. The two it does not serve, `Other edible` and `Other non-edible`,
appear only in the denominator of the feed share. A share computed without them therefore
divides by too small a total and comes out **too high**: it is an upper bound on nature's
contribution, not an estimate of it, and it cannot reproduce the author's figures.

`feed_lambda_by_country` handles this: it computes from whatever categories are present and
returns `lambda_is_upper_bound`, which is `True` whenever a category is missing. Do not strip
that column downstream.

The compiled extract (`gleam3_dmi.xlsx`) remains the ask. This harvest is the fallback.

## Why it cannot be scripted headlessly

The app is Shiny over a websocket, and it will not hand out a session to a plain HTTP client
(`RuntimeError: no session id`). It also suspends rendering for any client that reports itself
hidden, so a background or offscreen browser tab returns an empty table forever. The window has
to be genuinely visible while the harvest runs.

## Steps

1. Open <https://foodandagricultureorganization.shinyapps.io/GLEAMV3_Public/> in Chrome, and
   leave the window visible and frontmost for the whole run. Do not minimise it or switch to a
   different desktop.
2. Click **Input Data** in the left sidebar. Four sections appear: Herd Parameters, Manure
   Management Systems, Live Weights and **Dry Matter Intake**. They are empty until a country is
   selected, because the World-level table is not populated.
3. Pick any country in the **Area** selector at the top left and confirm the Dry Matter Intake
   table fills. Its columns should be `Area, Animal, LPS` followed by the six feed categories.
   One country selection returns every species and production system at once, so the harvest is
   about 231 steps rather than 231 times 6.
4. Open the browser console (View, Developer, JavaScript Console) and paste the script below.
   It walks every country, waits for that country's table to arrive before reading it, and
   accumulates the rows on `window.__gleam`.
5. Watch progress with `__gleam.done + " / " + __gleam.total`. Expect roughly fifteen minutes.
6. When `__gleam.finished` is true, run the download line at the bottom. It saves
   `gleam3_dmi_dashboard.csv` to your Downloads folder.
7. Move that file to `base_data/global_invest/livestock_provision/gleam3_dmi_dashboard.csv` and
   add its `es_parameters` row, then the feed share can be computed.

## The script

```javascript
(() => {
  const el = document.querySelector('#selectCountry');
  const options = el.selectize.options;
  const isAggregate = /^(WORLD|CONTINE|FAOREG|GLEAMR|SDGsRE|EU27|LDC|LLDC|SIDS|OECD|G20)/i;
  const countries = Object.keys(options)
    .filter(k => !isAggregate.test(k))
    .map(k => ({code: k, label: (options[k].label || options[k].text || '')}));

  window.__gleam = {rows: [], heads: null, done: 0, total: countries.length, empty: []};
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const readTable = () => {
    const t = document.querySelector('#inputDryMatterIntakeSummary table');
    if (!t) return null;
    return {
      heads: [...t.querySelectorAll('thead th')].map(x => x.innerText.trim()),
      body: [...t.querySelectorAll('tbody tr')].map(r => [...r.cells].map(c => c.innerText.trim())),
    };
  };

  (async () => {
    for (const c of countries) {
      el.selectize.setValue(c.code, false);
      Shiny.setInputValue('selectCountry', c.code, {priority: 'event'});
      let got = null;
      for (let i = 0; i < 40; i++) {                 // up to 16 s per country
        await sleep(400);
        const t = readTable();
        if (t && t.body.length && t.body[0][1] === c.label) { got = t; break; }
      }
      if (!got) { window.__gleam.empty.push(c.code); window.__gleam.done++; continue; }
      window.__gleam.heads = got.heads;
      for (const row of got.body) window.__gleam.rows.push([c.code, ...row]);
      window.__gleam.done++;
    }
    window.__gleam.finished = true;
    console.log('done', window.__gleam.rows.length, 'rows;',
                window.__gleam.empty.length, 'countries returned nothing');
  })();
  return 'harvest started';
})()
```

## The download line

```javascript
(() => {
  const g = window.__gleam;
  const esc = v => `"${String(v).replace(/"/g, '""')}"`;
  const csv = [['country_code', ...g.heads].map(esc).join(',')]
    .concat(g.rows.map(r => r.map(esc).join(','))).join('\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], {type: 'text/csv'}));
  a.download = 'gleam3_dmi_dashboard.csv';
  a.click();
  return g.rows.length + ' rows written';
})()
```

## Checking what came back

Countries listed in `__gleam.empty` returned no table. A handful is normal (small territories
GLEAM does not model). If most are empty, the window lost focus and the app stopped rendering:
bring it back to the front and run the harvest again.
