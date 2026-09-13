#!/usr/bin/env python3
"""Export the component inventory from the AppSheet app to a flat CSV.

Read-only: only the API "Find" action is used, nothing is written back.

Credentials are read from the environment or from ~/.config/mspc/appsheet.env
(never from the repository):

    APPSHEET_APP_ID=<app guid>
    APPSHEET_ACCESS_KEY=V2-...

Usage:
    tools/inventory/appsheet_export.py [-o inventory/stock.csv]
"""

import argparse
import csv
import json
import os
import pathlib
import sys
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.appsheet.com/api/v2/apps/{app}/tables/{table}/Action"
ENV_FILE = pathlib.Path.home() / ".config" / "mspc" / "appsheet.env"

COLUMNS = [
    "Category", "Label", "Value", "Unit", "Tolerance", "Power", "Description",
    "Package", "Case", "Termination", "Manufacturer", "Code", "Amount",
    "Reserved", "Available", "Box", "BarCode", "Datasheet", "Row ID",
]


def load_credentials():
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.startswith("#"):
                env[key.strip()] = value.strip()
    env.update({k: v for k, v in os.environ.items() if k.startswith("APPSHEET_")})
    try:
        return env["APPSHEET_APP_ID"], env["APPSHEET_ACCESS_KEY"]
    except KeyError as missing:
        sys.exit(f"missing {missing} (set it in the environment or {ENV_FILE})")


def find_all(app, key, table):
    url = API.format(app=app, table=urllib.parse.quote(table))
    body = json.dumps({"Action": "Find", "Properties": {}, "Rows": []}).encode()
    request = urllib.request.Request(url, data=body, method="POST", headers={
        "ApplicationAccessKey": key,
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        sys.exit(f"{table}: HTTP {error.code}: {error.read().decode(errors='replace')}")


def url_of(link):
    """AppSheet stores Url columns as a JSON object; keep just the address."""
    try:
        return json.loads(link).get("Url", "") if link else ""
    except (json.JSONDecodeError, AttributeError):
        return link


def to_int(text):
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", default="inventory/stock.csv")
    args = parser.parse_args()

    app, key = load_credentials()
    by_id = lambda table: {row["Row ID"]: row for row in find_all(app, key, table)}
    packages, units, boxes = by_id("Packages"), by_id("Units"), by_id("Box")
    items = find_all(app, key, "Items")

    rows = []
    for item in items:
        package = packages.get(item["Package"], {})
        amount, reserved = to_int(item["Amount"]), to_int(item["Reserved"])
        rows.append({
            **{column: item.get(column, "") for column in COLUMNS},
            "Unit": units.get(item["Unit"], {}).get("Unit", ""),
            "Package": package.get("Code", ""),
            "Case": package.get("Case", ""),
            "Box": boxes.get(item["Box"], {}).get("Name", ""),
            "Datasheet": url_of(item["Datasheet"]),
            "Amount": amount,
            "Reserved": reserved,
            "Available": amount - reserved,
        })
    rows.sort(key=lambda row: (row["Category"], row["Package"], row["Label"]))

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} items -> {output}")


if __name__ == "__main__":
    main()
