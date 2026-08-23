#!/usr/bin/env python3
"""Voice and structure gate for the GEP deck.

Every rule Chiara has set on overview.qmd, as an executable predicate, so that a
rule stated once fails the build forever after instead of being caught by hand:

    python3 docs/check_deck.py docs/overview.qmd

Exits non-zero if any rule fires.

What is masked before matching: the YAML front matter (it holds the numbers, not
prose), fenced code blocks, and the {{< meta ... >}} shortcodes (a shortcode name
is not prose). Everything else is matched on whitespace-normalised text, because a
sentence that wraps across two source lines otherwise defeats a naive pattern.

The structural rules (D3 to D8) encode the harmonisation Chiara asked for on
2026-08-22: one slide per service, every service slide opening with what the
service values, and every Q an actual question.
"""

import re
import sys
from pathlib import Path

# Slides that are not a single library service, so the "one service per slide"
# and "state line opens with Values" rules do not apply to them.
NON_SERVICE_SLIDES = {'marine_carbon'}

# A service slide's state line must open with this verb, so every slide answers
# "what does this value?" before anything else.
STATE_OPENER = 'Values '

# Prose rules: (id, description, pattern, what to do instead).
RULES = [
    ("D1", "self-referential status tag: describes the bookkeeping, not the claim",
     r"marked as such|documented as such|flagged accordingly|noted as such|"
     r"recorded as (an? )?(open item|such)|the data asks are closed|"
     r"as (noted|recorded|documented) above",
     "Delete it. Whether something is labelled somewhere is not actionable."),

    ("D2", "framing preamble or colon-label frame",
     r"[Tt]he (point|upshot|takeaway|key thing) (here )?is|"
     r"[Ii]t is (important|worth) (to note|noting|stressing)|"
     r"[Ww]hat this means is|[Ii]n other words",
     "Delete the frame, keep the content."),

    ("D3", "numeral-parallelism slogan",
     r"\b[Oo]ne\s+\w[\w\- ]{0,24},\s+(?:one|two|three|four)\s+\w|"
     r"\b[Tt]wo\s+\w[\w\- ]{0,24},\s+(?:one|two|three|four)\s+\w",
     "State the relationship in a sentence, or name the topic plainly."),

    ("D4", "aphorism equation",
     r"\b\w+ is (data|architecture|policy|the product|the interface)\b",
     "State the concrete mechanism instead."),

    ("D5", "em dash",
     r"—",
     "Use a comma, a full stop, or brackets."),

    ("D6", "hedging without content",
     r"\bit can be argued\b|\bgenerally considered\b|\bmay potentially\b|"
     r"\barguably\b|\bit should be noted\b",
     "Assert the fact, or drop the sentence."),

    ("D7", "editorialising self-validation",
     r"\bhonestly\b|\bgenuinely\b|\bcrucially\b|\bnotably\b|\brobustly\b|"
     r"\bimportantly\b",
     "The content must carry the credibility."),

    ("D8", "empty adjective pair",
     r"\bclear and concise\b|\bquick and easy\b|\bsimple and effective\b",
     "One word, or the fact."),

    # V-rules: read off Chiara's own edits to service_status.qmd on 2026-08-22 and 2026-08-23.
    # Every one of them cut a clause that explained a claim she had already made. What she did NOT
    # do is shorten: the length distribution is unchanged by her pass (median 20 words before and
    # after, longest 80 both), so there is deliberately no sentence-length rule here -- a long
    # sentence where every clause carries new content is hers to keep.
    # ", and which" was tried here and withdrawn: every hit was an indirect question
    # ("and which one is current is open"), not an add-on clause.
    ("V1", "trailing add-on clause",
     r", which also\b|, which additionally\b",
     "Cut it, or make it its own sentence with its own subject."),

    ("V2", "purpose tail explaining a change already named",
     r"\bso (it|they|these|those) could be \w+ed\b|\bso that (it|they) could be \w+ed\b|"
     r"\bin order to make (it|them) \w+able\b",
     "Name what moved. Why it moved is the commit message's job."),

    ("V3", "colon-label where a question is meant",
     r"\bThe (decision|question|open item) (what|whether|which|how)\b",
     "Ask it: 'What is X?' rather than 'The decision what X is:'."),

    ("V4", "filler quantifier",
     r"\bon top\.|\bon top of that\b|\bas well\.|\bto boot\b",
     "Delete it, or fold the item into the list it belongs to."),

    ("V5", "hedge tail on an absence",
     r"\bto compare against yet\b|\bnot (yet )?available at this (stage|point)\b|"
     r"\bfor the time being\b",
     "State the absence. 'No reference output' already means we do not have one."),

    ("V6", "unit written out where the short form reads better",
     r"\bper kilogram\b|\bper hectare of\b|\bper square kilometre\b",
     "per kg, per ha, per km2."),
]


def strip_front_matter(text):
    """The YAML header holds the deck's numbers, not its prose, so it is not matched."""
    if text.startswith('---'):
        end = text.find('\n---', 3)
        if end != -1:
            return text[end + 4:]
    return text


def mask(text):
    """Fenced code and meta shortcodes replaced by blanks of the same length."""
    text = re.sub(r'```.*?```', lambda m: ' ' * len(m.group(0)), text, flags=re.S)
    text = re.sub(r'\{\{<[^>]*>\}\}', lambda m: ' ' * len(m.group(0)), text)
    return text


def slides(body):
    """(heading, slide text) per `##` slide, plus the `#` section each sits under."""
    out, section, heading, buf = [], '', None, []
    for line in body.split('\n'):
        if line.startswith('## '):
            if heading is not None:
                out.append((section, heading, '\n'.join(buf)))
            heading, buf = line[3:].strip(), []
        elif line.startswith('# '):
            if heading is not None:
                out.append((section, heading, '\n'.join(buf)))
                heading, buf = None, []
            section = line[2:].strip()
        elif heading is not None:
            buf.append(line)
    if heading is not None:
        out.append((section, heading, '\n'.join(buf)))
    return out


def check_prose(name, body):
    """The word-level rules, over whitespace-normalised masked text."""
    flat = re.sub(r'\s+', ' ', mask(body))
    hits = []
    for rid, desc, pattern, fix in RULES:
        for m in re.finditer(pattern, flat):
            hits.append((rid, desc, name, flat[max(0, m.start() - 70):m.start() + 70], fix))
    return hits


def check_structure(name, body):
    """The harmonisation rules: one service per slide, a state line that opens with
    what the service values, questions that are questions, and bold used only as a label."""
    hits = []
    for section, heading, text in slides(body):
        is_service_slide = section.startswith('Open questions by service')
        slug = heading.split()[0] if heading else ''

        if is_service_slide and ' and ' in heading:
            hits.append(("D9", "two services shared one slide", name, heading,
                         "One slide per service."))

        if is_service_slide and slug not in NON_SERVICE_SLIDES:
            state = re.search(r'\[(.+?)\]\{\.state\}', text, flags=re.S)
            if not state:
                hits.append(("D10", "service slide has no state line", name, heading,
                             f"Open with a [{STATE_OPENER}...]{{.state}} line."))
            elif not re.sub(r'\s+', ' ', state.group(1)).strip().startswith(STATE_OPENER):
                opening = re.sub(r'\s+', ' ', state.group(1))[:70]
                hits.append(("D11", "state line does not open with what the service values",
                             name, f'{heading}: {opening}',
                             f"Start it with '{STATE_OPENER}...'."))

        for line in text.split('\n'):
            q = re.match(r'\s*-\s+\*\*(Q\d+):\*\*\s*(.+)', line)
            if q:
                body_text = mask(q.group(2))
                if '?' not in body_text:
                    hits.append(("D12", "a Q bullet that asks nothing", name,
                                 f'{heading} {q.group(1)}: {body_text[:80]}',
                                 "Make it one interrogative sentence, or move it to the sheet."))
                elif body_text.index('?') > 320:
                    hits.append(("D13", "the ask arrives too late in a Q bullet", name,
                                 f'{heading} {q.group(1)}: {body_text[:80]}',
                                 "Put the question first, then at most one supporting sentence."))

            # Bold is a label device here: **Qn:**, **No open questions**, or a
            # leading **Label:** on a list item. Mid-sentence bold is emphasis.
            stripped = line.strip()
            for m in re.finditer(r'\*\*(.+?)\*\*', stripped):
                allowed = (re.match(r'\*\*Q\d+:\*\*', stripped[m.start():])
                           or m.group(1).startswith('No open questions')
                           or (m.start() <= 2 and m.group(1).rstrip().endswith(':')))
                if not allowed:
                    hits.append(("D14", "bold used for emphasis rather than as a label",
                                 name, f'{heading}: {m.group(1)[:70]}',
                                 "Remove the bold; the sentence must carry the weight."))
    return hits


def check_tables(name, body):
    """Every table gets the same {.smaller} treatment, so they render alike."""
    hits = []
    for section, heading, text in slides(body):
        if re.search(r'^\|.*\|$', text, flags=re.M) and '{.smaller}' not in heading:
            hits.append(("D15", "table slide without {.smaller}", name, heading,
                         "Add {.smaller} to the heading, like every other table."))
    return hits


def check_literal_numbers(name, raw, body):
    """A figure that already lives in the YAML must be referenced, never retyped,
    or the two drift apart the next time the number changes."""
    header = raw[:raw.index('\n---', 3)] if raw.startswith('---') else ''
    values = {}
    for m in re.finditer(r'^(\w+):\s*"([^"]+)"', header, flags=re.M):
        value = m.group(2)
        # Only distinctive figures. A bare small integer (a count like "3" or "23")
        # occurs all over ordinary prose, so matching it reports the rule's own noise
        # rather than a retyped number.
        if re.search(r'[$%]|\d[.,]\d|\d\s*(bn|T|M|m3)\b', value):
            values[value] = m.group(1)
    flat = re.sub(r'\s+', ' ', mask(body))
    hits = []
    for literal, key in values.items():
        if literal in flat:
            hits.append(("D16", "a YAML figure retyped as a literal", name,
                         f'{literal} (is {{{{< meta {key} >}}}})',
                         "Reference the metadata so one edit updates every slide."))
    return hits


def main():
    paths = [Path(a) for a in sys.argv[1:]] or [Path('docs/overview.qmd')]
    failures = []
    for path in paths:
        raw = path.read_text()
        body = strip_front_matter(raw)
        failures += check_prose(path.name, body)
        # The structure, table and literal-number rules describe a slide deck. The voice rules
        # above apply to any page Chiara writes, so the status page is checked for those only.
        if path.name == 'overview.qmd':
            failures += (check_structure(path.name, body) + check_tables(path.name, body)
                         + check_literal_numbers(path.name, raw, body))

    print('=' * 78)
    print('DECK GATE -- ' + ', '.join(p.name for p in paths))
    print('=' * 78)
    if not failures:
        print('\nCLEAN: every rule holds.\n')
        return 0

    by_rule = {}
    for rid, desc, fname, ctx, fix in failures:
        by_rule.setdefault((rid, desc, fix), []).append((fname, ctx))
    for (rid, desc, fix), hits in sorted(by_rule.items()):
        print(f'\n[{rid}] {desc}  ({len(hits)})')
        print(f'      -> {fix}')
        for fname, ctx in hits[:8]:
            print(f'      {fname}: ...{ctx.strip()}...')
        if len(hits) > 8:
            print(f'      ... and {len(hits) - 8} more')
    print(f'\n{len(failures)} violation(s) across {len(by_rule)} rule(s).\n')
    return 1


if __name__ == '__main__':
    sys.exit(main())
