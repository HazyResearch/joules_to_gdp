# Wages and occupation data

Use these public source snapshots to match tasks with occupations and hourly pay. Their SHA-256 checksums are pinned in the [input manifest](../reproduction/input_manifest.json).

| File | Source and role |
| --- | --- |
| `data/input/OEWS.csv` | U.S. Bureau of Labor Statistics (BLS), May 2024 Occupational Employment and Wage Statistics. The valuation module reads national cross-industry employment and mean/median hourly wages from this file. |
| `data/input/ONET.csv` | O*NET® OnLine “All Occupations” CSV export, used as an occupation/code list in the local occupation-data workflow. |
| `data/input/task_statements.csv` | O*NET® task statements, used as occupation/task source text in that workflow. |

The exact O*NET® release numbers for these local snapshots were not recorded. Their original bytes and checksums are preserved. The valuation calculations read `OEWS.csv` directly. The O*NET® files provide occupational source metadata for inspecting task mappings. GDPVal prompts, recorded expert completion times, and model judgments are separate inputs.

## Task valuation

**Output dollar value = recorded expert completion time × occupation median hourly wage.** A win or tie against the selected expert reference receives this value credit.

1. Supply each task's occupation mapping and recorded expert completion time using the [input manifest](../reproduction/input_manifest.json).
2. [Verify and stage the inputs](../reproduction/README.md#verify-external-inputs). The validator also checks these bundled source files.
3. Follow [Run calculations](../analysis/README.md#run-calculations) to join tasks with wages and calculate output dollar values.

The calculations read wages directly from `OEWS.csv`. Occupation weights support summaries across the selected task panel. Store calculated values and model scores in ignored output directories. The source snapshots contain neither recorded expert completion times nor calculated task values.

## Sources and attribution

The BLS [May 2024 OEWS tables](https://www.bls.gov/oes/tables.htm) provide the wage source. BLS [publishes its data in the public domain](https://www.bls.gov/opub/copyright-information.htm) and asks for source credit. The O*NET® occupation export comes from [O*NET OnLine](https://www.onetonline.org/find/all), and task statements come from the [O*NET Database](https://www.onetcenter.org/database.html). Their [OnLine](https://www.onetonline.org/help/license) and [Database](https://www.onetcenter.org/license_db.html) content licenses require attribution under CC BY 4.0. Credit: U.S. Department of Labor, Employment and Training Administration, O*NET® program. O*NET® is its trademark; these local copies are not endorsed by the Department of Labor.
