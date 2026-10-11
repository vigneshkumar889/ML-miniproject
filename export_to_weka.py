"""Export the salary CSV dataset to Weka's ARFF format."""

import argparse
import csv
import math
from pathlib import Path


def quote_arff(value: str) -> str:
    """Quote an ARFF name or nominal value."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )
    return f"'{escaped}'"


def is_numeric(values: list[str]) -> bool:
    if not values:
        return False
    try:
        return all(math.isfinite(float(value)) for value in values)
    except ValueError:
        return False


def export_csv_to_arff(source: Path, destination: Path) -> int:
    with source.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames
        if not fieldnames or any(not name for name in fieldnames):
            raise ValueError("The CSV must have a non-empty header row.")
        if len(set(fieldnames)) != len(fieldnames):
            raise ValueError("The CSV contains duplicate column names.")
        if "salary" not in fieldnames:
            raise ValueError("The CSV must contain a 'salary' target column.")

        rows = list(reader)

    if not rows:
        raise ValueError("The CSV contains no data rows.")

    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"CSV row {row_number} has the wrong number of fields.")

    columns = [name for name in fieldnames if name != "salary"] + ["salary"]
    values_by_column = {
        column: [(row[column] or "").strip() for row in rows]
        for column in columns
    }
    numeric_columns = {
        column: is_numeric([value for value in values_by_column[column] if value])
        for column in columns
    }
    if not numeric_columns["salary"]:
        raise ValueError("The 'salary' target column must contain numeric values.")

    lines = ["@relation salary_prediction", ""]
    for column in columns:
        if numeric_columns[column]:
            lines.append(f"@attribute {quote_arff(column)} NUMERIC")
            continue

        categories = list(dict.fromkeys(
            value for value in values_by_column[column] if value
        ))
        if not categories:
            raise ValueError(f"Column '{column}' has no non-missing values.")
        nominal_values = ",".join(quote_arff(value) for value in categories)
        lines.append(f"@attribute {quote_arff(column)} {{{nominal_values}}}")

    lines.extend(["", "@data"])
    for row_values in zip(*(values_by_column[column] for column in columns)):
        arff_values = []
        for column, value in zip(columns, row_values):
            if not value:
                arff_values.append("?")
            elif numeric_columns[column]:
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"Column '{column}' contains a non-finite number.")
                arff_values.append(value)
            else:
                arff_values.append(quote_arff(value))
        lines.append(",".join(arff_values))

    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(rows)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Convert a salary CSV file to Weka's ARFF format."
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=project_dir / "salarydata.csv",
        help="Input CSV (default: salarydata.csv)",
    )
    parser.add_argument(
        "destination",
        nargs="?",
        type=Path,
        default=project_dir / "salarydata.arff",
        help="Output ARFF (default: salarydata.arff)",
    )
    args = parser.parse_args()
    row_count = export_csv_to_arff(args.source, args.destination)
    print(f"Exported {row_count} rows to {args.destination}")


if __name__ == "__main__":
    main()
