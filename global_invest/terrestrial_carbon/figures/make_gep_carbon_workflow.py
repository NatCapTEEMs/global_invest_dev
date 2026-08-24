"""Draw GEP_carbon_workflow.png for terrestrial_carbon_method.qmd (run from anywhere; writes next
to itself). A repo doc asset, not a ProjectFlow task: the diagram is static (no run data flows in),
so regenerating it belongs with the docs, not in the per-run tree.

Three phases, matching the implemented calculation (terrestrial_carbon_tasks.py):
(a) data preparation: Spawn 2010 density + base-year ESA LULC + carbon zones -> density lookup table
(b) application: lookup applied to base-year LULC x zones -> per-cell density -> x cell area -> stock
(c) valuation: sum by country (r250) -> x annual rental SCC -> GEP by country
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

GREEN = '#1a5632'
GRAY_FILL = '#f2f2f2'
GRAY_EDGE = '#888888'
INK = '#222222'

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GEP_carbon_workflow.png')

fig, ax = plt.subplots(figsize=(14, 6.2))
ax.set_xlim(0, 14); ax.set_ylim(0, 6.2); ax.axis('off')


def box(x, y, w, h, text, fill=GRAY_FILL, edge=GRAY_EDGE, textcolor=INK, fontsize=10.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08,rounding_size=0.12',
                                facecolor=fill, edgecolor=edge, linewidth=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', color=textcolor,
            fontsize=fontsize, fontweight='bold' if bold else 'normal', linespacing=1.35)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=16,
                                 color='#555555', linewidth=1.6, shrinkA=2, shrinkB=2))


# phase bands + titles
for x0, x1, label in [(0.15, 5.20, '(a) Data preparation'),
                      (5.40, 9.70, '(b) Application'),
                      (9.90, 13.90, '(c) Valuation')]:
    ax.add_patch(FancyBboxPatch((x0, 0.25), x1 - x0, 5.3, boxstyle='round,pad=0.02,rounding_size=0.1',
                                facecolor='none', edgecolor='#cccccc', linewidth=1.0, linestyle=(0, (4, 3))))
    ax.text(x0 + 0.15, 5.75, label, ha='left', va='center', color=GREEN, fontsize=13, fontweight='bold')

# (a) inputs
box(0.30, 4.05, 3.0, 1.0, 'Spawn et al. (2020) harmonized\ncarbon density 2010 (Mg C ha$^{-1}$,\nabove- + belowground)', fontsize=9.5)
box(0.30, 2.60, 3.0, 1.0, 'ESA CCI LULC, 300 m\n(base year 2019)', fontsize=9.5)
box(0.30, 1.15, 3.0, 1.0, 'Carbon zones\n(Gibbs & Ruesch 2008)', fontsize=9.5)
# (a) lookup
box(3.60, 2.30, 1.45, 1.6, 'Carbon density\nlookup table\n\nmean Mg C ha$^{-1}$\nper (LULC class\n× carbon zone)', fill='white', edge=GREEN, fontsize=9)
arrow(3.30, 4.55, 3.60, 3.65)
arrow(3.30, 3.10, 3.60, 3.10)
arrow(3.30, 1.65, 3.60, 2.55)

# (b) application
box(5.60, 2.30, 2.0, 1.6, 'Per-cell carbon density\n(Mg C ha$^{-1}$)\n\nlookup applied to the\nbase-year LULC ×\ncarbon zone maps', fill='white', edge=GREEN, fontsize=9)
box(7.90, 2.30, 1.6, 1.6, 'Per-cell carbon\nstock (Mg C)\n\n× cell area\n(ha per cell)', fill='white', edge=GREEN, fontsize=9)
arrow(5.05, 3.10, 5.60, 3.10)
arrow(7.60, 3.10, 7.90, 3.10)
ax.text(7.55, 1.15, 'the same chain applies to any LULC time series\n(scenario / annual maps) for dynamic assessment',
        ha='center', va='center', color='#666666', fontsize=8.8, style='italic')

# (c) valuation
box(10.10, 2.30, 1.75, 1.6, 'Carbon stock\nby country\n\nsum per country\n(one row per\ncountry, r250)', fill='white', edge=GREEN, fontsize=9)
box(12.10, 2.30, 1.65, 1.6, 'GEP by country\n(2019 USD)\n\n× annual rental SCC\n(Parisa et al. 2022;\nRennert et al. 2022)', fill=GREEN, edge=GREEN, textcolor='white', fontsize=8.7)
arrow(9.50, 3.10, 10.10, 3.10)
arrow(11.85, 3.10, 12.10, 3.10)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
print('wrote', OUT)
