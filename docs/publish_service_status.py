#!/usr/bin/env python3
"""Publish docs/service_status.csv to the group's shared status sheet.

The CSV in this repo is the source of truth: it is version controlled, so a change to any
service's status shows up in a diff beside the code change that caused it. The Google Sheet is a
rendering of it for the group, refreshed by running this.

    python3 docs/publish_service_status.py

It writes to the SAME sheet every time (the group's link must keep working), so the file id is
fixed here rather than passed in. Column widths and per-row heights are set from the content,
because Google keeps a one-line height on rows it has seen before and a wrapped cell then
renders clipped however the wrap flag is set.
"""
import math
import os
import subprocess
import sys
import tempfile

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'service_status.csv')
SHEET_REMOTE = 'gdrive:gep/gep_library_status.xlsx'
SHEET_TITLE = 'library status'
# Column widths in characters, in the CSV's column order.
COLUMN_WIDTHS = (22, 16, 22, 11, 8, 64, 16, 64, 40)
LINE_HEIGHT_POINTS = 13.5
MAX_LINES = 14


def build_workbook(df):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = SHEET_TITLE
    sheet.append(list(df.columns))
    for row in df.itertuples(index=False):
        sheet.append(['' if pd.isna(v) else v for v in row])

    for index, width in enumerate(COLUMN_WIDTHS, start=1):
        sheet.column_dimensions[sheet.cell(1, index).column_letter].width = width
    for cell in sheet[1]:
        cell.font = Font(bold=True)

    for row_index in range(1, sheet.max_row + 1):
        lines = 1
        for column_index, width in enumerate(COLUMN_WIDTHS, start=1):
            text = str(sheet.cell(row_index, column_index).value or '')
            if text:
                lines = max(lines, math.ceil(len(text) / width))
        sheet.row_dimensions[row_index].height = round(
            min(lines, MAX_LINES) * LINE_HEIGHT_POINTS + 4, 1)
        for column_index in range(1, len(COLUMN_WIDTHS) + 1):
            sheet.cell(row_index, column_index).alignment = Alignment(
                wrap_text=True, vertical='top')
    return workbook


def main():
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'gep_library_status.xlsx')
        build_workbook(df).save(path)
        result = subprocess.run(
            ['rclone', 'copyto', path, SHEET_REMOTE, '--drive-import-formats', 'xlsx'],
            capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f'rclone failed with code {result.returncode}')
    print(f'published {len(df)} services to {SHEET_REMOTE}')


if __name__ == '__main__':
    main()
