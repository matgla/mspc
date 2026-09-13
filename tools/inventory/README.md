# Inventory

Boards in this repository can be imported as AppSheet projects with
[ElectronicStorageUtils](https://github.com/matgla/ElectronicStorageUtils):

```sh
esu gui --config tools/inventory/esu-projects.json
esu import-bom --config tools/inventory/esu-projects.json --project "MSPC mainboard v3"   # dry run
```

- `esu-projects.json` lists the boards and their root schematics.
- `bom_matches.json` (created on first save) records which stock item each BOM line
  uses, so re-importing after a schematic change is automatic. Commit it.
