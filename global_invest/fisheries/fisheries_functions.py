"""Fisheries ES science: read the pre-computed marine-fisheries shock headers.

Marine-fisheries productivity is mapped by RCP (DBEM vs BOATS provenance is a paper question, see task
#16) and is never derived from the terrestrial SEALS maps, so unlike carbon and pollination it is READ
from a pre-computed HAR (cwon_shocks.har, headers FI26/FI45/FI85, one per RCP) rather than recomputed.

The headers carry a FULL ANNUAL series (Y2017..Y2050, 50 regions x 34 years). For the current file the
series is a 2017->2018 step that then holds flat, so it is already constant over the 2023-2050 solve
window; the read returns the whole annual series regardless, so the task reads each year directly and a
genuinely dynamic future source (DBEM/Fish-MIP, #45) needs no read change.
"""


def read_fisheries_headers(cwon_path, headers):
    """Per-region ANNUAL fisheries shock (%) per RCP header -> {header: {reg: {year_int: value}}}."""
    from gtappy.harpy.har_file import HarFileObj
    h = HarFileObj(filename=cwon_path)
    out = {}
    for hdr in headers:
        arr = h[hdr].array
        regs = [s.strip() for s in h[hdr].setElements[0]]
        years = [int(s.strip().lstrip('Yy')) for s in h[hdr].setElements[1]]
        out[hdr] = {reg: {yr: arr[i, j] for j, yr in enumerate(years)}
                    for i, reg in enumerate(regs)}
    return out
