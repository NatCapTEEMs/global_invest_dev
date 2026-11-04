"""Read heartbeat.tsv of a running shares job and say whether it is progressing. Takes no arguments.

For a login node while the job runs. Progress is CPU time advancing between the last two samples,
since the task logs only a few lines in ten hours. Flags a stall when CPU time has not advanced,
memory is within a tenth of the 48 GB request, or the last sample is more than 25 minutes old.

Run: python -I -B run_shares_heartbeat_check.py
"""
import csv
import datetime
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MEM_KB = 48 * 1024 * 1024


def cpu_seconds(value):
    """ps 'times' is seconds already; older ps prints [[dd-]hh:]mm:ss."""
    if value.isdigit():
        return int(value)
    days, _, clock = value.rpartition('-')
    parts = [int(x) for x in clock.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    return (int(days) if days else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def main():
    path = os.path.join(HERE, 'heartbeat.tsv')
    if not os.path.exists(path):
        print('no heartbeat.tsv: the job has not started')
        return 1
    with open(path, newline='') as f:
        rows = [r for r in csv.DictReader(f, delimiter='\t') if r['cpu_s']]
    if not rows:
        print('heartbeat.tsv has no sample yet')
        return 1
    last = rows[-1]
    age = (datetime.datetime.now(datetime.timezone.utc)
           - datetime.datetime.strptime(last['utc'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc)).total_seconds()
    rss = int(last['rss_kb'])
    print('last sample %s, %.0f min ago: elapsed %.2f h, CPU %.2f h, memory %.1f GB'
          % (last['utc'], age / 60, int(last['elapsed_s']) / 3600, cpu_seconds(last['cpu_s']) / 3600, rss / 1048576))
    problems = []
    if len(rows) > 1 and cpu_seconds(last['cpu_s']) <= cpu_seconds(rows[-2]['cpu_s']):
        problems.append('CPU time did not advance since the previous sample')
    if rss > 0.9 * MEM_KB:
        problems.append('memory within a tenth of the 48 GB request')
    if age > 25 * 60:
        problems.append('no sample for %.0f minutes (finished, or the node is gone)' % (age / 60))
    print('STALL: ' + '; '.join(problems) if problems else 'progressing')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
