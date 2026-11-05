"""Gates for the library's prevention_shares at the EPS-5 threshold on the original stacks.

The run files in the folder above call these. Each gate stops with Mismatch on the first thing that
differs from what was pinned. Nothing here writes a file. Gates shared with the repaired-stack run
come from that package's committed module, so the repaired stacks enter only as a refusal.
"""
import csv
import json
import re


class Mismatch(Exception):
    """Something differs from what was pinned. The run stops on the first one."""


def read_rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def read_parameters(path):
    return {r['parameter']: r['value'] for r in read_rows(path)}


def setting_value(text):
    """A pinned setting as the value it stands for: true and false as booleans, the rest as numbers."""
    return {'true': True, 'false': False}[text] if text in ('true', 'false') else float(text)


def check_author_settings(manifest, settings):
    """Stop unless the author's run manifest holds every pinned setting that it records."""
    for row in settings:
        key = row['author_manifest_key']
        if key and manifest.get(key) != setting_value(row['value']):
            raise Mismatch('the author manifest has %s = %s, the pin is %s' % (key, manifest.get(key), row['value']))


def check_task_settings(p, settings):
    """Stop unless the task will read every pinned setting, after hydration."""
    for row in settings:
        found = getattr(p, row['task_attribute'], None)
        if found != setting_value(row['value']):
            raise Mismatch('%s is %r on the project, the pin is %s' % (row['task_attribute'], found, row['value']))


def flatten(value, prefix=''):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(item, prefix + key + '.'))
        return out
    return {prefix[:-1]: value}


def manifest_differences(eps5, ses11):
    """The dotted keys on which the two author manifests differ."""
    a, b = flatten(eps5), flatten(ses11)
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def check_manifests_differ_only_by_threshold(eps5, ses11):
    """The EPS-5 and SES-11 runs must share inputs and settings, and differ in the high threshold, outputs and time."""
    unexpected = [k for k in manifest_differences(eps5, ses11)
                  if k not in ('timestamp_utc', 'elapsed_minutes', 'threshold_high_t_ha_yr') and not k.startswith('outputs.')]
    if unexpected:
        raise Mismatch('the EPS-5 and SES-11 author manifests also differ in %s' % unexpected)


def policy_at_threshold(policy_bytes, old='11.0', new='5.0'):
    """A threshold policy CSV with its default-high rows moved to another rate, every other byte kept, and how many moved."""
    return re.subn((',%s,default-high' % re.escape(old)).encode(), (',%s,default-high' % new).encode(), policy_bytes)


def refuse_repaired(paths):
    """The comparison is on the original stacks. Any path into the repair stops it."""
    for path in paths:
        if 'repaired' in str(path) or 'spam_stack_repair' in str(path):
            raise Mismatch('%s belongs to the repaired stacks, which stay out of this run' % path)


def check_original_stacks(rows, pins):
    """Stop unless the library's pinned stacks and band map are the bytes the author's run read."""
    by_parameter = {r['parameter']: r for r in rows}
    by_role = {r['role']: r['sha256'] for r in pins}
    for parameter, role in (('erosion_area_stack_path', 'area_stack'), ('erosion_yield_stack_path', 'yield_stack'), ('erosion_bandmap_csv_path', 'bandmap')):
        refuse_repaired([by_parameter[parameter]['path']])
        if by_parameter[parameter]['sha256'] != by_role[role]:
            raise Mismatch('%s has SHA-256 %s, the author pin %s' % (parameter, by_parameter[parameter]['sha256'], by_role[role]))


def check_run_manifest(path, settings, named_inputs):
    """After the run: its own record must hold the pinned settings and name the inputs it was meant to read."""
    with open(path, encoding='utf-8') as f:
        recorded = json.load(f)
    check_author_settings(recorded, settings)
    for key, expected in named_inputs.items():
        if recorded['inputs'][key] != expected:
            raise Mismatch('the prevention-share manifest records %s = %s, not %s' % (key, recorded['inputs'][key], expected))
    refuse_repaired(recorded['inputs'].values())
    return recorded
