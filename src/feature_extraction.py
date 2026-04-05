import decimal
import re
import json
import argparse
from pathlib import Path

import pandas as pd
import sqlglot
import sqlglot.expressions as exp
from sqlglot.errors import ParseError, TokenError

# Priority ordering of dialects to speed up validation
DIALECTS_TO_TRY = [
    None, 'mysql', 'postgres', 'oracle', 'tsql', 'sqlite',
    'bigquery', 'snowflake', 'redshift', 'databricks',
    'hive', 'spark', 'spark2', 'presto', 'trino', 'athena',
    'clickhouse', 'duckdb', 'teradata', 'exasol', 'druid',
    'dremio', 'drill', 'solr', 'tableau', 'dune', 'prql',
    'doris', 'fabric', 'materialize', 'risingwave', 'singlestore', 'starrocks'
]

def clean_waf_artifacts(payload):
    """Replaces known WAF evasion fuzzing clusters from the dataset with a safe dummy value."""
    patterns_to_fix = [
        r'\+\\\.', r'\\\.\<\\', r'%!\<@', r'\\\<\\', r'\\#',
        r'!\<@', r'\<@\.\.', r'\<@\.\$', r'\$\s\.', r'\\\.+', r'\"\"\"\"\<@'
    ]
    for pattern in patterns_to_fix:
        payload = re.sub(pattern, ' 1 ', payload)
    return payload

def extract_sql_features(payload):
    """Parses the payload through multiple contexts and dialects to extract AST features."""
    if not isinstance(payload, str):
        return _empty_feature_dict()

    clean_payload = payload.strip('"').strip()
    clean_payload = clean_waf_artifacts(clean_payload)

    contexts = [
        f"{clean_payload}",
        f"SELECT * FROM users WHERE 1=1 {clean_payload}",
        f"SELECT * FROM users WHERE name = 'dummy{clean_payload}",
        f'SELECT * FROM users WHERE name = "dummy{clean_payload}',
        f"SELECT * FROM users WHERE 1=1 {clean_payload}'",
        f"SELECT * FROM users WHERE name = 'dummy{clean_payload}'",
        f"SELECT * FROM users WHERE (name = 'dummy{clean_payload}')",
        f"SELECT * FROM users WHERE (1=1 {clean_payload})",
        f"SELECT * FROM users; {clean_payload}",
        f"SELECT * FROM users WHERE name = 'dummy'; {clean_payload}"
    ]

    for dialect in DIALECTS_TO_TRY:
        for index, query in enumerate(contexts):
            try:
                parsed_statements = sqlglot.parse(query, read=dialect)
                tables, literal_types, select_arm_widths, node_set = [], [], [], set()
                col_type_map = {}

                for statement in parsed_statements:
                    if statement is None:
                        continue
                    for table in statement.find_all(exp.Table):
                        if table.name and table.name not in tables:
                            tables.append(table.name)
                    for literal in statement.find_all(exp.Literal):
                        if literal.args.get("is_string"):
                            literal_types.append("TEXT")
                        else:
                            literal_types.append("INTEGER")
                    for column in statement.find_all(exp.Column):
                        if column.name and column.name not in col_type_map:
                            idx = len(col_type_map)
                            col_type_map[column.name] = literal_types[idx] if idx < len(literal_types) else "TEXT"
                    for select in statement.find_all(exp.Select):
                        if select.parent and type(select.parent) is exp.Union:
                            select_arm_widths.append(len(select.expressions))
                    for node in statement.walk():
                        node_set.add(type(node).__name__)

                return {
                    "is_valid_syntax": True,
                    "winning_context_index": int(index),
                    "winning_dialect": dialect if dialect else "default",
                    "tables": json.dumps(tables),
                    "columns": json.dumps(list(col_type_map.keys())),
                    "literal_types": json.dumps(list(col_type_map.values())),
                    "select_arm_widths": json.dumps(select_arm_widths),
                    "node_set": json.dumps(sorted(node_set))
                }

            except (ParseError, TokenError, ValueError, decimal.InvalidOperation):
                continue
            except Exception:
                continue

    return _empty_feature_dict()

def _empty_feature_dict():
    return {
        "is_valid_syntax": False,
        "winning_context_index": -1,
        "winning_dialect": None,
        "tables": "[]",
        "columns": "[]",
        "literal_types": "[]",
        "select_arm_widths": "[]",
        "node_set": "[]"
    }

def main():
    parser = argparse.ArgumentParser(description="Extract AST features from generated payloads.")
    parser.add_argument("--input", required=True, help="Path to input generated_payloads.csv")
    parser.add_argument("--output", required=True, help="Path to output Feature_Extraction_Results.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Could not find {input_path}")
        return

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    
    # Standardize column name to 'Query' so test_sandbox logic works seamlessly
    target_col = 'payload' if 'payload' in df.columns else 'Query'
    if target_col not in df.columns:
        print(f"Error: Could not find 'payload' or 'Query' column in {input_path}")
        return

    print("Extracting AST features (this may take a moment)...")
    features_df = df[target_col].apply(extract_sql_features).apply(pd.Series)
    
    # Merge and rename to ensure compatibility with sandbox
    out_df = pd.concat([df, features_df], axis=1)
    if 'Query' not in out_df.columns and 'payload' in out_df.columns:
        out_df['Query'] = out_df['payload']

    out_df.to_csv(output_path, index=False)
    print(f"Successfully saved {len(out_df)} rows to {output_path}")

if __name__ == "__main__":
    main()