#!/usr/bin/env python3
"""Check a KiCad BOM against the inventory snapshot and list what to order.

Inputs:
  - the schematic (BOM exported with kicad-cli, one row per reference),
  - a part map (mainboard/bom/part_map.csv) saying which stock row or MPN each
    value+footprint uses; a row with `refs` overrides the value+footprint row
    for those references,
  - the inventory snapshot from appsheet_export.py (inventory/stock.csv).

Usage:
    tools/inventory/bom_check.py [--boards N] [--spare PCT] [-o report.md]
"""

import argparse
import collections
import csv
import math
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCES = ("stock", "substitute", "verify", "order", "none")


def export_bom(schematic):
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "bom.csv"
        subprocess.run([
            "kicad-cli", "sch", "export", "bom", str(schematic), "-o", str(out),
            "--fields", "Reference,Value,Footprint,${DNP}",
            "--labels", "Reference,Value,Footprint,DNP",
            "--group-by", "", "--exclude-dnp",
        ], check=True, capture_output=True)
        return list(csv.DictReader(out.open()))


def part_key(value, footprint):
    """Match value+footprint loosely: '20mΩ' == '20m', and the hand-solder or
    pad-size variants of a footprint count as the same land pattern."""
    footprint = footprint.split(":")[-1]
    footprint = re.sub(r"_Pad[0-9.x]+mm", "", footprint)
    footprint = re.sub(r"_HandSolder(ing)?$", "", footprint, flags=re.IGNORECASE)
    if re.fullmatch(r"[0-9.]+[kmMR]?Ω?", value):
        value = value.replace("Ω", "").rstrip("R")
    return value, footprint


def resolve_stock(stock, prefix):
    hits = [row for row in stock if row["Row ID"].startswith(prefix)]
    if len(hits) != 1:
        sys.exit(f"stock_id {prefix!r} matches {len(hits)} inventory rows")
    return hits[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--schematic", default=ROOT / "mainboard/mainboard/mainboard.kicad_sch")
    parser.add_argument("--map", default=ROOT / "mainboard/bom/part_map.csv")
    parser.add_argument("--stock", default=ROOT / "inventory/stock.csv")
    parser.add_argument("--boards", type=int, default=1)
    parser.add_argument("--spare", type=float, default=10,
                        help="extra percent on ordered passives (0402/0603/0805/1206)")
    parser.add_argument("-o", "--output", help="write the report here instead of stdout")
    args = parser.parse_args()

    stock = list(csv.DictReader(open(args.stock)))
    by_key, by_ref = {}, {}
    for row in csv.DictReader(open(args.map)):
        if row["source"] not in SOURCES:
            sys.exit(f"unknown source {row['source']!r} for {row['value']}")
        if row["refs"]:
            for ref in row["refs"].split(","):
                by_ref[ref.strip()] = row
        else:
            by_key[part_key(row["value"], row["footprint"])] = row

    lines = collections.OrderedDict()
    unmapped = []
    for part in export_bom(args.schematic):
        footprint = part["Footprint"].split(":")[-1]
        row = by_ref.get(part["Reference"]) or by_key.get(part_key(part["Value"], footprint))
        if row is None:
            unmapped.append(f"{part['Reference']} {part['Value']} {footprint}")
            continue
        if row["source"] == "none":
            continue
        key = row["stock_id"] or row["mpn"]
        line = lines.setdefault(key, {"row": row, "refs": [], "values": set()})
        line["refs"].append(part["Reference"])
        line["values"].add(part["Value"])

    have, order, check = [], [], []
    for key, line in lines.items():
        row, need = line["row"], len(line["refs"]) * args.boards
        item = resolve_stock(stock, row["stock_id"]) if row["stock_id"] else None
        available = int(item["Available"]) if item else 0
        refs = ",".join(line["refs"])
        label = f"{item['Label']} ({item['Box']})" if item else ""
        if row["source"] == "order" or available < need:
            short = need - (available if row["source"] in ("stock", "substitute") else 0)
            passive = any(size in row["footprint"] for size in ("0402", "0603", "0805", "1206"))
            buy = math.ceil(short * (1 + args.spare / 100)) if passive else short
            order.append((row, need, buy, refs, label))
        if row["source"] in ("stock", "substitute") and available >= need:
            have.append((row, need, available, refs, label))
        if row["source"] == "verify":
            check.append((row, need, available, refs, label))

    out = [f"# BOM vs inventory — {pathlib.Path(args.schematic).name}, {args.boards} board(s)", ""]
    if unmapped:
        out += ["## Not in the part map", ""] + [f"- {text}" for text in unmapped] + [""]
    out += ["## To order", "", "| buy | need | MPN | manufacturer | LCSC | requirement | refs | note |",
            "|---:|---:|---|---|---|---|---|---|"]
    for row, need, buy, refs, _ in sorted(order, key=lambda o: o[0]["mpn"]):
        out.append(f"| {buy} | {need} | {row['mpn']} | {row['manufacturer']} | {row['lcsc']} | "
                   f"{row['requirement']} | {refs} | {row['note']} |")
    out += ["", "## Check before using stock", "", "| need | have | stock item | fallback MPN | refs | note |",
            "|---:|---:|---|---|---|---|"]
    for row, need, available, refs, label in check:
        out.append(f"| {need} | {available} | {label} | {row['mpn']} | {refs} | {row['note']} |")
    out += ["", "## From stock", "", "| need | have | stock item | MPN | refs | note |",
            "|---:|---:|---|---|---|---|"]
    for row, need, available, refs, label in have:
        note = ("substitute: " if row["source"] == "substitute" else "") + row["note"]
        out.append(f"| {need} | {available} | {label} | {row['mpn']} | {refs} | {note} |")

    text = "\n".join(out) + "\n"
    if args.output:
        pathlib.Path(args.output).write_text(text)
        print(f"{len(order)} to order, {len(check)} to check, {len(have)} from stock -> {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
