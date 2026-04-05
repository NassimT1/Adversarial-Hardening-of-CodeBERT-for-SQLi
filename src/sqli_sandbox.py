# SQLi detection sandbox

#  -------------- DESCRIPTION: -------------- 
# Cleans and prepares the input data, sets up a temporary database, executes the query, and checks for possible attacks
# Processes complete queries, query fragments, and even malformed SQL
# Detects common injection attacks such as tautologies, UNION attacks, blind attacks including boolean and time-based blind attacks, stacked query attacks, and obfuscations
# --------------------------------------------

# -------------- FILE STRUCTURE -------------- 
# 1. Constants: This includes the canary value, SQL keywords, and regex patterns.
# 2. Data structures: This includes classes such as ASTProfile, DatabaseBlueprint, and SandboxResult.
# 3. Normalization: This includes functions that normalize the input text.
# 4. Blueprint builders: This is the logic that determines the structure of the fake database schema.
# 5. Database builder: This is the code that creates the in-memory SQLite database and populates it with trap data.
# 6. Canary checks: This includes the code that checks if there are any secret tokens exposed.
# 7. Static filters: This includes the list of regex patterns and keywords that are used to identify common attacks.
# 8. Static analysis: This includes the functions that check the query for common attacks.
# 9. Fragment templates: This includes the sample queries that are used to test incomplete SQL code.
# 10. Behavioral detection: This is the logic that checks the execution results to determine if there is exploitation.
# 11. Sandbox class: This is the main SQLiSandbox class.
# ---------------------------------------------

import re
import sqlite3
import urllib.parse

# --- 1. Constants ---

CANARY_INT = 999_999_777
CANARY_STR = "__CANARY_EXFIL__"
CANARY_SECRET = "CANARY_TOKEN_7f4a9c"

SQL_KW = (
    "ON|WHERE|SET|JOIN|LEFT|RIGHT|INNER|OUTER|FULL|CROSS|NATURAL|"
    "HAVING|GROUP|ORDER|LIMIT|UNION|EXCEPT|INTERSECT|AS|AND|OR|NOT|"
    "IN|EXISTS|BETWEEN|LIKE|IS|NULL|CASE|WHEN|THEN|ELSE|END|BY|ASC|"
    "DESC|INSERT|INTO|VALUES|UPDATE|DELETE|CREATE|DROP|ALTER|SELECT|"
    "FROM|WITH|DISTINCT|ALL|TOP|OFFSET|FETCH"
)

TABLE_RE = re.compile(
    r"\b(?:FROM|JOIN|INTO|UPDATE|TABLE)\s+([\w\[\]`\"\.]+)"
    r"(?:\s+(?:AS\s+)?(?!(?:" + SQL_KW + r")\b)([\w]+))?",
    re.IGNORECASE,
)

FROM_MULTI_RE = re.compile(
    r"\bFROM\s+([\w\[\]`\"\.]+(?:\s*,\s*[\w\[\]`\"\.]+)+)",
    re.IGNORECASE,
)


# --- 2. Data structures ---

class ASTProfile:
    # Created by get_ast_profile()
    # The context index tells us if it's a full query (0), a fragment (1-9), or broken (-1)
    def __init__(
        self,
        is_valid=False,
        winning_context_index=-1,
        winning_dialect=None,
        tables=None,
        columns=None,
        literal_types=None,
        select_arm_widths=None,
        node_set=None
    ):
        self.is_valid = is_valid
        self.winning_context_index = winning_context_index
        self.winning_dialect = winning_dialect
        self.tables = tables if tables is not None else []
        self.columns = columns if columns is not None else []
        self.literal_types = literal_types if literal_types is not None else []
        self.select_arm_widths = select_arm_widths if select_arm_widths is not None else []
        self.node_set = node_set if node_set is not None else set()


class DatabaseBlueprint:
    # Instructions for building the sandbox database, like table layouts and how we'll catch data leaks
    # Canary strategies include secrets tables, canary rows, and arithmetic triggers
    def __init__(
        self,
        tables=None,
        union_arm_width=0,
        canary_strategy="standard",
        augment_text_canary=False,
        expect_cross_db_error=False,
        needs_correlated_table=False
    ):
        self.tables = tables if tables is not None else {}
        self.union_arm_width = union_arm_width
        self.canary_strategy = canary_strategy
        self.augment_text_canary = augment_text_canary
        self.expect_cross_db_error = expect_cross_db_error
        self.needs_correlated_table = needs_correlated_table


class SandboxResult:
    def __init__(
        self,
        malicious,
        exploit_type,
        mode,
        input_text,
        executed_sql,
        rows_returned,
        inferred_schema,
        detection_reason,
        error=None,
        executed=False
    ):
        self.malicious = malicious
        self.exploit_type = exploit_type
        self.mode = mode
        self.input_text = input_text
        self.executed_sql = executed_sql
        self.rows_returned = rows_returned
        self.inferred_schema = inferred_schema
        self.detection_reason = detection_reason
        self.error = error
        self.executed = executed

    def __repr__(self):
        flag = "MALICIOUS" if self.malicious else "BENIGN   "
        cols = {t: list(c.keys()) for t, c in self.inferred_schema.items()}
        return (
            f"[{flag}] type={self.exploit_type or 'none':<26} "
            f"executed={self.executed}\n"
            f"           reason={self.detection_reason}\n"
            f"           schema={cols}"
        )


# --- 3. Normalization ---

def normalize(text):
    # Clean up the text by decoding URLs, HTML entities, and hex, then strip the comments out
    try:
        text = urllib.parse.unquote(text)
    except Exception:
        pass

    for entity, char in [
        ("&apos;", "'"), ("&#39;", "'"), ("&#x27;", "'"),
        ("&quot;", '"'), ("&#34;", '"'), ("&#x22;", '"'),
        ("&amp;",  "&"), ("&#38;", "&"), ("&lt;", "<"), ("&gt;", ">"),
    ]:
        text = text.replace(entity, char)

    text = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), text)

    def decode_hex(m):
        try:
            raw = bytes.fromhex(m.group(1))
        except ValueError:
            return m.group(0)
        if len(raw) < 4:
            return m.group(0)
        try:
            if all(raw[i] == 0 for i in range(1, len(raw), 2)):
                return raw[::2].decode("ascii", errors="replace")
            return raw.decode("ascii", errors="replace")
        except Exception:
            return m.group(0)

    text = re.sub(r"\b0x([0-9a-fA-F]{8,})\b", decode_hex, text)
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"--[^\n]*", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_literals(text):
    # Empty out text inside quotes so we don't accidentally flag normal words
    text = re.sub(r"'[^']*'", "''", text)
    return re.sub(r'"[^"]*"', '""', text)


def strip_comments(sql):
    sql = re.sub(r"--[^\n]*",  " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"#[^\n]*",   " ", sql)
    return re.sub(r"\s+", " ", sql).strip()


# --- 4. Blueprint builders ---

ARITHMETIC_NODES = {"Mod", "Div", "If", "Case", "Floor", "Rand", "Exp"}
CONCAT_NODES = {"Chr", "DPipe", "Concat", "Hex", "HexString", "Elt", "Repeat"}
SUBQUERY_NODES = {"Subquery"}
CROSS_DB_NODES = {"GenerateSeries", "Parameter"}
UNION_NODES = {"Union"}


def build_blueprint_from_profile(profile):
    # Turn the AST profile into a database blueprint that we can actually build
    bp = DatabaseBlueprint()

    if profile.tables:
        col_type_map = dict(zip(profile.columns, profile.literal_types))

        # Match columns to their tables if they have a dot. Otherwise, share them across tables
        pinned = {t: {} for t in profile.tables}
        unattributed = {}

        for col in profile.columns:
            typ = col_type_map.get(col, "TEXT")
            if "." in col:
                tname, cname = col.split(".", 1)
                matched = next((t for t in profile.tables if t.lower() == tname.lower()), None)
                if matched:
                    pinned[matched][cname] = typ
                else:
                    unattributed[col] = typ
            else:
                unattributed[col] = typ

        for table_name in profile.tables:
            cols = dict(pinned[table_name])
            if not cols:
                cols.update(unattributed)
            if profile.node_set & ARITHMETIC_NODES and not any(t == "INTEGER" for t in cols.values()):
                cols["_num"] = "INTEGER"
            if not cols:
                cols["_id"] = "INTEGER"
            bp.tables[table_name] = cols

    else:
        # We don't have table details, so just use a standard users table
        bp.tables = {"users": {"id": "INTEGER", "username": "TEXT", "password": "TEXT", "role": "TEXT"}}

    if UNION_NODES & profile.node_set and profile.select_arm_widths:
        bp.union_arm_width = profile.select_arm_widths[0]

    node_set = profile.node_set
    if UNION_NODES & node_set:
        bp.canary_strategy = "secrets_table"
    elif ARITHMETIC_NODES & node_set:
        bp.canary_strategy = "arithmetic_trigger"
    elif SUBQUERY_NODES & node_set:
        bp.canary_strategy = "correlated_table"
        bp.needs_correlated_table = True
    elif {"Or", "And", "EQ"} & node_set:
        bp.canary_strategy = "canary_rows"
    else:
        bp.canary_strategy = "standard"

    if CONCAT_NODES & node_set:
        bp.augment_text_canary = True
    if CROSS_DB_NODES & node_set:
        bp.expect_cross_db_error = True

    return bp


def build_blueprint_from_template(template_schema, node_set, select_arm_widths):
    # Blueprint for incomplete queries. The layout comes from our template, not the query profile
    bp = DatabaseBlueprint()
    bp.tables = {t: dict(c) for t, c in template_schema.items()}

    if UNION_NODES & node_set and select_arm_widths:
        bp.union_arm_width = select_arm_widths[0]
    elif UNION_NODES & node_set and bp.tables:
        bp.union_arm_width = len(next(iter(bp.tables.values())))
    else:
        bp.union_arm_width = 0

    if UNION_NODES & node_set:
        bp.canary_strategy = "secrets_table"
    elif ARITHMETIC_NODES & node_set:
        bp.canary_strategy = "arithmetic_trigger"
    elif SUBQUERY_NODES & node_set:
        bp.canary_strategy = "correlated_table"
        bp.needs_correlated_table = True
    elif {"Or", "And", "EQ"} & node_set:
        bp.canary_strategy = "canary_rows"
    else:
        bp.canary_strategy = "standard"

    if CONCAT_NODES & node_set:
        bp.augment_text_canary = True
    if CROSS_DB_NODES & node_set:
        bp.expect_cross_db_error = True

    return bp


# --- 5. Database builder ---

def qi(name):
    return '"' + name.replace('"', '""') + '"'


def build_db_from_blueprint(blueprint):
    # Spin up a temporary SQLite database in memory and insert our trap data
    conn = sqlite3.connect(":memory:")

    # Make sure the secrets table exists so we can catch UNION attacks trying to read it
    conn.execute("CREATE TABLE secrets (id INTEGER PRIMARY KEY, token TEXT NOT NULL)")
    conn.execute(f"INSERT INTO secrets VALUES (1, '{CANARY_SECRET}')")

    for table_name, columns in blueprint.tables.items():
        if table_name.lower() == "secrets":
            continue

        seen = {}
        for col, typ in columns.items():
            seen.setdefault(col.lower(), (col, typ))
        deduped = {c: t for c, t in seen.values()}

        if not deduped:
            deduped = {"_id": "INTEGER"}

        if blueprint.union_arm_width > 0:
            current_cols = list(deduped.items())
            target = blueprint.union_arm_width
            if len(current_cols) < target:
                for i in range(target - len(current_cols)):
                    deduped[f"__pad{i}__"] = "TEXT"
            elif len(current_cols) > target:
                deduped = dict(current_cols[:target])

        col_defs = ", ".join(f"{qi(c)} {t}" for c, t in deduped.items())
        col_names = list(deduped.keys())
        quoted = ", ".join(qi(c) for c in col_names)
        phs = ", ".join("?" * len(col_names))
        ins_sql = f"INSERT INTO {qi(table_name)} ({quoted}) VALUES ({phs})"

        conn.execute(f"CREATE TABLE IF NOT EXISTS {qi(table_name)} ({col_defs})")

        if blueprint.canary_strategy == "arithmetic_trigger":
            for i, seed in enumerate([42, 100, 255]):
                row = [seed if deduped[c] == "INTEGER" else f"_seed_{chr(97+i)}_" for c in col_names]
                try: conn.execute(ins_sql, row)
                except Exception: pass
            canary_row = [CANARY_INT if deduped[c] == "INTEGER" else CANARY_STR for c in col_names]
            try: conn.execute(ins_sql, canary_row)
            except Exception: pass

        elif blueprint.canary_strategy in ("secrets_table", "canary_rows", "standard"):
            for i in range(3):
                row = [
                    [7_000_001, 7_000_002, 7_000_003][i] if deduped[c] == "INTEGER" else f"_seed_{chr(97+i)}_"
                    for c in col_names
                ]
                try: conn.execute(ins_sql, row)
                except Exception: pass
            canary_row = [CANARY_INT if deduped[c] == "INTEGER" else CANARY_STR for c in col_names]
            try: conn.execute(ins_sql, canary_row)
            except Exception: pass

        elif blueprint.canary_strategy == "correlated_table":
            for i in range(2):
                row = [[1, 2][i] if deduped[c] == "INTEGER" else f"_seed_{chr(97+i)}_" for c in col_names]
                try: conn.execute(ins_sql, row)
                except Exception: pass
            canary_row = [CANARY_INT if deduped[c] == "INTEGER" else CANARY_STR for c in col_names]
            try: conn.execute(ins_sql, canary_row)
            except Exception: pass

        # Add another trap row for text merging attacks unless we already did it above
        if blueprint.augment_text_canary and blueprint.canary_strategy not in (
            "secrets_table", "canary_rows", "standard", "correlated_table",
        ):
            text_canary_row = [CANARY_INT if deduped[c] == "INTEGER" else CANARY_STR for c in col_names]
            try: conn.execute(ins_sql, text_canary_row)
            except Exception: pass

    if blueprint.needs_correlated_table:
        conn.execute("CREATE TABLE IF NOT EXISTS _corr (id INTEGER PRIMARY KEY, val TEXT NOT NULL)")
        conn.execute(f"INSERT INTO _corr VALUES (1, '{CANARY_STR}')")
        conn.execute("INSERT INTO _corr VALUES (2, '_normal_')")

    conn.commit()
    return conn


# --- 6. Canary checks ---

def canary_in_rows(rows):
    for row in rows:
        for cell in row:
            s = str(cell)
            if s in (CANARY_STR, CANARY_SECRET, str(CANARY_INT)):
                return s
    return None


def secrets_canary_in_rows(rows):
    return any(CANARY_SECRET in str(cell) for row in rows for cell in row)


# --- 7. Static filters ---

CROSS_DB_SIGNALS = [
    "utl_http", "utl_inaddr", "sys.all_tables", "dba_role_privs",
    "sys.loginuser", "sys.user$", "all_tab_columns",
    "dbms_pipe.", "dbms_ldap.",
    "all_users", "all_tables",
    "v$version", "global_name",
    "@@version", "@@datadir", "@@basedir", "@@hostname", "@@global.",
    "mysql.user", "mysql.db", "mysql.host", "information_schema",
    "sys_eval(", "sys_exec(", "sys.eval(", "sys.exec(",
    "soname",
    "xp_cmdshell", "xp_regread",
    "waitfor delay", "waitfor ",
    "is_srvrolemember", "sysusers", "sysobjects", "syscolumns",
    "pg_sleep(", "pg_read_file(", "pg_ls_dir(", "pg_catalog",
    "pg_tables", "pg_user",
    "generate_series(",
    "rdb$fields", "rdb$", "sysibm.", "sysibm.systables", "syscat.",
    "make_set(", "elt(", "updatexml(", "extractvalue(", "char(",
    "isnull(", "benchmark(", "sleep(",
    "load_file(", "into outfile", "into dumpfile",
    "iif(",
    "exec(", "sp_executesql(",
    "sp_", "xp_",
    "randomblob(",
    "domain.domains", "domain.columns", "domain.tables",
    "bfilename", "identified by", "user_name(",
    "execute immediate",
    "declare @",
]

CROSS_DB_ERRORS = [
    "no such table: mysql.", "no such table: rdb$", "no such table: sysibm",
    "no such table: pg_",   "no such table: domain.", "no such table: sys.",
    "no such table: information_schema",
    "no such table: all_tables", "no such table: all_tab", "no such table: dual",
    "no such function: generate_series", "no such function: nvl",
    "no such function: decode", "no such function: charindex",
    "no such function: sleep", "no such function: benchmark",
    "no such function: pg_sleep", "no such function: waitfor",
    "no such function: make_set", "no such function: updatexml",
    "no such function: extractvalue", "no such function: load_file",
    "no such function: utl_http", "no such function: dbms_",
    'unrecognized token: "@"',
    "near \"waitfor\"",
    "near \"exec\"",
    "near \"execute\"",
]

INJECTION_ERRORS_COMPLETE = [
    "division by zero",
    "order by term out of range",
]

BLIND_RE = re.compile(r"\bcase\s+when\s*\(?\s*-?\d+\s*[=<>!]{1,2}\s*-?\d+", re.IGNORECASE)
ORDER_PROBE_RE = re.compile(r"\bORDER\s+BY\s+\d+\s*(?:[#\-]{1,2})", re.IGNORECASE)
QUOTE_COMMENT_RE = re.compile(r"""(?:'(?!\w)|["`])[^'"`\n]{0,30}\s*(?:--|/\*|\#)""", re.IGNORECASE)
SQL_GATE_RE = re.compile(
    r"""
    \b(?:union|select|insert|update|delete|exec(?:ute)?|
         sleep|benchmark|waitfor|declare|convert|
         truncate|alter|procedure|grant|revoke|
         hex|unhex|iif|bfilename|load_file|
         ascii|substring|substr|concat|char|ord)
    (?:\s|\(|$|,)
    | \b(?:or|and)\s+\(?\s*(?:\d|['"]|\w+\s*\(|\w+\s*[=!<>]|\w+\s+(?:like|in|between|is)\b|(?:having|where|select|union|null|not|true|false|exists|like|in|between|is)\b|\bnot\s+\w)
    | \blike\s+[\'\"']
    | \bnull\b
    | -- | /\* | \x23(?!\w) | @@ | 0x[0-9a-fA-F]{6,} | \d+\s*[=<>!]=?\s*\d+
    """,
    re.IGNORECASE | re.VERBOSE,
)
SELECT_TOP_FRAG_RE = re.compile(r"^\s*\(?\s*select\s+top\b", re.IGNORECASE)

STACKED_KW_RE = re.compile(
    r"(?i)^(?:SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|EXECUTE|EXEC|"
    r"WITH|GRANT|REVOKE|REPLACE|CALL|PRAGMA|ATTACH|DETACH|BEGIN|COMMIT|ROLLBACK|"
    r"DECLARE|PRINT|SP_\w+|XP_\w+|SET|USE|GO|"
    r"COPY|DO|PREPARE|DEALLOCATE|FETCH|CLOSE|MOVE|LOCK|LISTEN|NOTIFY|UNLISTEN|"
    r"VACUUM|ANALYZE|ANALYSE|CLUSTER|REINDEX|CHECKPOINT|EXPLAIN|DISCARD|RESET|SHOW|"
    r"SAVEPOINT|RELEASE|RAISE|RETURN|MERGE|LOAD|"
    r"FLUSH|KILL|REPAIR|OPTIMIZE|CHECK|HANDLER|INSTALL|UNINSTALL|SIGNAL|RESIGNAL|"
    r"BACKUP|RESTORE|BULK|SHUTDOWN|RECONFIGURE|WAITFOR"
    r")(?:\s|$|[(])",
)

MSSQL_IF_RE = re.compile(r"\bif\s*\(\s*\d+\s*=\s*\d+\s*\)\s*select\b", re.IGNORECASE)
BOOL_PROBE_RE = re.compile(r"(?:\|\||\\+)\s*\(\s*select\s+['\"\\w]+", re.IGNORECASE)
OR_EXISTS_RE = re.compile(r"\bor\s+\(?\s*(?:exists|not\s+exists)\b", re.IGNORECASE)
OR_IN_SUBQ_RE = re.compile(r"\b(?:or|and)\s+\S+\s+in\s*\(\s*select\b", re.IGNORECASE)
AND_LIKE_SAME_RE = re.compile(r"\b(?:and|end\s+and)\s+(?:\s*\(\s*)*[\"'](\w+)[\"'](?:\s*\)\s*)*\s+like\s+[\"']\1", re.IGNORECASE)
PRINT_VAR_RE = re.compile(r"\bprint\s+@@", re.IGNORECASE)
DUAL_PROBE_RE = re.compile(r"\bfrom\s+dual\s+where\s+\d+\s*=\s*\d+", re.IGNORECASE)
UNBAL_QUOTE_BOOL_RE = re.compile(r"^\s*-?\d[\d%]*[%'\"'][\s)]*(?:and|or)\b", re.IGNORECASE)
SQLMAP_FALSE_PROBE_RE = re.compile(r"\b(?:and|or)\s+\(?\s*(-?\d{3,})\s*[=*]\s*(\d{3,})", re.IGNORECASE)
CHAR_PIPE_CONCAT_RE = re.compile(r"\bchar\s*\(\s*\d+\s*\)\s*\|\|", re.IGNORECASE)
OR_EMPTY_NOISE_RE = re.compile(r"\bor\s+''\s*(?:[^=a-zA-Z0-9\s\w]|'(?!\w))", re.IGNORECASE)
END_AND_TRUNCATED_RE = re.compile(r"\bend\s+and\s+(?:\s*\(\s*)*['\"'][\w%]*['\"'](?:\s*\)\s*)*\s*[=<>]", re.IGNORECASE)
IN_BOOLEAN_MODE_RE = re.compile(r"\bin\s+boolean\s+mode\b", re.IGNORECASE)
SLEEP_NOPAREN_RE = re.compile(r"(?:^\s*[\$]?\s*\(?\s*|[^a-z])sleep\s+\d", re.IGNORECASE)
ADV_SUBQ_PROBE_RE = re.compile(r"\(SELECT\s*[\w\s*,]+\)\s*[^a-zA-Z(]{0,5}(?:=|!=|<>|LIKE|BETWEEN)", re.IGNORECASE)
OR_NOISE_SUBQ_RE = re.compile(r"\bOR\s*[^a-zA-Z(]{0,8}\(\s*SELECT\b", re.IGNORECASE)
OR_IDENT_LIKE_IDENT_RE= re.compile(r"\bOR\s+([a-zA-Z_]\w{0,20})\s+LIKE\s+\1\b", re.IGNORECASE)
EXECUTE_IMMEDIATE_RE = re.compile(r"\bexecute\s+immediate\b", re.IGNORECASE)

STANDALONE_SQL_KW = frozenset({
    "select", "insert", "update", "delete", "drop", "create", "alter",
    "truncate", "exec", "execute", "union", "having", "replace", "procedure",
    "print", "distinct", "grant", "revoke", "commit", "rollback", "declare",
    "like", "order", "where", "from", "join", "begin",
})


# --- 8. Static analysis ---

def has_stacked_query(sql, require_context=False):
    stripped = strip_literals(sql)
    for m in re.finditer(r";", stripped):
        after = re.sub(r"^/\*.*?\*/\s*", "", stripped[m.end():].strip(), flags=re.DOTALL).strip()
        if after and STACKED_KW_RE.match(after):
            if not require_context:
                return True
            before = stripped[:m.start()]
            has_unbal_quote = (before.count("'") % 2 == 1 or before.count('"') % 2 == 1)
            has_comment = bool(re.search(r"(?:--|#|/\*)", stripped[m.end():]))
            if has_unbal_quote or has_comment:
                return True
    return False


def has_cross_db(sql):
    lower = re.sub(r"\s*\(\s*", "(", re.sub(r"\s+", " ", sql).lower())
    for sig in CROSS_DB_SIGNALS:
        if "(" in sig:
            fn_name = sig.split("(")[0]
            if re.search(r"\b" + re.escape(fn_name) + r"\(", lower): return True
            if re.search(r"\b" + re.escape(fn_name) + r"[^a-zA-Z\s]{0,3}\s*\(", lower): return True
        elif sig.endswith("_"):
            if re.search(r"\b" + re.escape(sig) + r"\w*", lower): return True
            prefix = sig.rstrip("_")
            if re.search(r"\bexec\s+" + re.escape(prefix) + r"\b", lower): return True
        elif sig in lower:
            return True
    if re.match(r"^\s*use\s+\w+\s*$", lower): return True
    if re.search(r"\bcall\s+\w[\w.]*[^a-zA-Z\s]{0,3}\s*\(", lower): return True
    if re.search(r"\bv\$\w", lower): return True
    if re.search(r"@@\w", lower): return True
    return False


def has_blind_injection(text):
    return bool(BLIND_RE.search(normalize(text)))

def has_order_by_probe(text):
    return bool(ORDER_PROBE_RE.search(urllib.parse.unquote(text)))

def has_quote_comment(text):
    for m in QUOTE_COMMENT_RE.finditer(text):
        # Ignore this if it's just finishing a LIKE search pattern
        pre = text[:m.start()]
        if re.search(r"\blike\s+'[^']*$", pre, re.IGNORECASE):
            continue
        return True
    return False


def has_always_true(text):
    text = re.sub(r'""', '"', text)
    norm = normalize(text).lower().lstrip()

    for q in ('"', "'", "`"):
        qi_idx = norm.find(q)
        if 0 <= qi_idx <= 10:
            rest = norm[qi_idx + 1:].lstrip()
            if rest[:2] == "or" and (len(rest) == 2 or not rest[2].isalnum()):
                return True

    OP = r"(?:\s*\(\s*)*"
    CP = r"(?:\s*\)\s*)*"

    if re.search(r"\b(?:where|and|or)\s+" + OP + r"(\d{3,})\s*=\s*\1" + CP, norm): return True
    if re.search(r"^\s*" + OP + r"(\d{3,})\s*=\s*\1" + CP, norm): return True
    if re.search(r"\bor\s+" + OP + r"'([^']*)'" + CP + r"\s*=\s*" + OP + r"'?\1", norm): return True
    if re.search(r'\bor\s+' + OP + r'"([^"]*)"' + CP + r'\s*=\s*' + OP + r'"?\1', norm): return True
    if re.search(r"\band\s+" + OP + r"'([\w%]+)'" + CP + r"\s*=\s*" + OP + r"'?\1", norm): return True
    if re.search(r'\band\s+' + OP + r'"([\w%]+)"' + CP + r'\s*=\s*' + OP + r'"?\1', norm): return True
    if re.search(r"\band\s+" + OP + r"'([\w%]+)'" + CP + r"\s+like\s+" + OP + r"'?\1", norm): return True
    if re.search(r'\band\s+' + OP + r'"([\w%]+)"' + CP + r'\s+like\s+' + OP + r'"?\1', norm): return True
    if re.search(r"""\bor\s+'([^']*)'\\s*=\\s*n'\\1'""", norm): return True
    if re.search(r"""\bor\s+n'([^']*)'\\s*=\\s*'?\\1""", norm): return True
    if re.search(r"\bor\s+\w+\s+is\s+not\s+null\b", norm): return True
    if re.search(r"""\bor\s+('[^']*'|"[^"]*"|\w+)\s+in\s*\(\s*\1\s*\)""", norm): return True

    lower = strip_literals(norm)

    or_patterns = [
        r"\bor\s+1\s*=\s*1\b", r"\bor\s+true\b", r"\bor\s+not\s+false\b",
        r"\bor\s+-?\d+\s*=\s*-?\d+\b", r"\bor\s+-?\d+\s*[><]\s*-?\d+\b",
        r"\bor\s+ascii\(\d+\)\s*=\s*\d+\b",
        r"\bor\s+''\s*=\s*''", r'\bor\s+""\s*=\s*""', r"\bor\s+not\s+\d+\s*=\s*\d+",
    ]
    if any(re.search(p, lower) for p in or_patterns): return True

    for m in re.finditer(r"\b(?:and|having)\s+(-?\d+)\s*=\s*(-?\d+)\b", lower):
        if m.group(1) == m.group(2): return True

    if re.search(r"\bor\s+" + OP + r"([a-z_]\w*)\s*" + CP + r"=\s*" + OP + r"\1\b", lower): return True
    if re.search(r"\bor\s+'[^']*'\s*(?:>|>=|like)\s*'[^']*'", lower): return True

    m = re.search(r"\bor\s+(-?\d+)\s+between\s+(-?\d+)\s+and\s+(-?\d+)\b", lower)
    if m and int(m.group(2)) <= int(m.group(1)) <= int(m.group(3)): return True

    if re.search(r"\bor\s+\w+\s+like\s+'%'?", lower): return True
    if re.search(r"\bor\s+0x[1-9a-fA-F][0-9a-fA-F]*\b", lower): return True
    if re.search(r"\bor\s+true\b", lower): return True

    return False


def has_union_injection(text):
    # Catch comment-split UNION before we strip the comments out
    if re.search(r'uni(?:\s*/\*[^*]*\*/\s*|\s)+on\s+sel(?:\s*/\*[^*]*\*/\s*|\s)+ect\b', text, re.IGNORECASE):
        return True
    clean = re.sub(r"/\*!(?:\d+)?(.*?)\*/", r" \1 ", text, flags=re.DOTALL)
    clean = re.sub(r"\s+", " ", re.sub(r"/\*.*?\*/", "", clean, flags=re.DOTALL))
    if re.search(r"\bUNION\s+(?:ALL\s+)?SELECT\b", clean, re.IGNORECASE): return True
    if re.search(r"(?<![a-zA-Z])UNION[^a-zA-Z\s]{0,4}(?:ALL[^a-zA-Z\s]{0,4})?SELECT", clean, re.IGNORECASE): return True
    if re.search(r"\bUNION\s+ALL[^a-zA-Z\s]{1,4}SELECT", clean, re.IGNORECASE): return True
    return False


def has_injection_probe(text):
    norm = normalize(text)
    if MSSQL_IF_RE.search(norm): return True, "blind_boolean"
    if BOOL_PROBE_RE.search(norm): return True, "blind_boolean"
    if DUAL_PROBE_RE.search(norm): return True, "blind_boolean"
    if OR_EXISTS_RE.search(norm): return True, "tautology"
    if OR_IN_SUBQ_RE.search(norm): return True, "tautology"
    if AND_LIKE_SAME_RE.search(norm): return True, "tautology"
    if PRINT_VAR_RE.search(norm): return True, "cross_db_probe"
    if UNBAL_QUOTE_BOOL_RE.search(norm): return True, "blind_boolean"
    if CHAR_PIPE_CONCAT_RE.search(norm): return True, "encoding_obfuscation"
    if OR_EMPTY_NOISE_RE.search(norm): return True, "tautology"
    if END_AND_TRUNCATED_RE.search(norm): return True, "tautology"
    if IN_BOOLEAN_MODE_RE.search(norm): return True, "blind_boolean"
    if SLEEP_NOPAREN_RE.search(norm): return True, "blind_time"
    if ADV_SUBQ_PROBE_RE.search(norm): return True, "nested_injection"
    if OR_NOISE_SUBQ_RE.search(norm): return True, "nested_injection"
    if OR_IDENT_LIKE_IDENT_RE.search(norm): return True, "tautology"
    if EXECUTE_IMMEDIATE_RE.search(norm): return True, "stacked_queries"
    if norm.strip().lower() in STANDALONE_SQL_KW:
        return True, "cross_db_probe"

    m = SQLMAP_FALSE_PROBE_RE.search(norm)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if abs(a) != b: return True, "blind_boolean"

    m2 = re.search(r"^\s*\(?\s*(-?\d{3,})\s*=\s*(\d{3,})\s*\)", norm)
    if m2:
        a, b = int(m2.group(1)), int(m2.group(2))
        if abs(a) != b and abs(a) > 100 and b > 100: return True, "blind_boolean"

    if re.search(r"\bor\s+[1-9]\d*\s*(?:/\*|#|$)", norm, re.IGNORECASE): return True, "tautology"
    if re.match(r"^\s*end\s*(?:#|--)", norm, re.IGNORECASE): return True, "blind_boolean"
    if re.search(r"\bor\s+\(?[\"'][^\"']*[\"'][\"']?\s*=\s*[\"']", norm, re.IGNORECASE): return True, "tautology"
    if re.search(r"\buni\s+on\s+(?:all\s+)?select\b", norm, re.IGNORECASE): return True, "union_based"
    if re.search(r"@\w+\s+select\b", norm, re.IGNORECASE): return True, "cross_db_probe"
    if re.search(r"@(?:select|table|version|identity|rowcount|error)\b", norm, re.IGNORECASE): return True, "cross_db_probe"
    
    # Look for attacks calling functions with double pipes
    if re.search(r'\|\|\s*\w+\.\w+\s*\(', norm):
        return True, "nested_injection"
    
    # Look for a stray comment end followed by an attack pattern
    if re.search(r'\*/\s*(?:=\s*\d+|(?:and|or)\b)', norm, re.IGNORECASE): 
        return True, "comment_obfuscation"
    return False, ""


LEGACY_TO_CATEGORY = {
    "cross_db_probe": "cross_db_probe",
    "tautology": "tautology",
    "union_exfiltration": "union_based",
    "blind_injection": "blind_boolean",
    "auth_bypass": "comment_obfuscation",
    "order_by_probe": "blind_boolean",
    "error_based": "blind_boolean",
    "stacked_query": "stacked_queries",
    "comment_injection": "comment_obfuscation",
    "hex_encoding": "encoding_obfuscation",
    "nested_injection": "nested_injection",
}

def canonical(raw_type):
    if raw_type is None:
        return None
    return LEGACY_TO_CATEGORY.get(raw_type, raw_type)

def has_injection_context(text):
    # Checks for signs like broken quotes that mean someone is trying to break out of a string
    # This helps prevent false alarms
    if text.count("'") % 2 == 1:
        return True
    if re.search(r"--|/\*|#(?!\w)", text):
        return True
    if re.search(r"\bor\b\s+(?:\d|'|\")", text, re.IGNORECASE):
        return True
    return False


def run_static_filters(text, mode="static_only", skip_sql_gate=False, as_fallback=False):
    # Check patterns from most to least obvious. The first match triggers it, otherwise it's safe
    norm = normalize(text)
    stripped = text.strip()

    def hit(exploit_type, reason):
        return SandboxResult(
            malicious=True, exploit_type=canonical(exploit_type), mode=mode,
            input_text=text, executed_sql="", rows_returned=[], inferred_schema={},
            detection_reason=reason, executed=False,
        )

    # For stacked queries, make sure it looks like an attack so we don't flag normal multi-statement scripts
    if has_stacked_query(text, require_context=True):
        return hit("stacked_queries", "semicolon-delimited second statement detected")
    if has_cross_db(norm): return hit("blind_time", "cross-DB function or system object reference")
    if has_blind_injection(text): return hit("blind_boolean", "CASE WHEN numeric blind injection pattern")
    if has_always_true(text): return hit("tautology", "always-true condition detected")
    if has_order_by_probe(text): return hit("blind_boolean", "ORDER BY N column-count enumeration probe")
    if has_quote_comment(text): return hit("comment_obfuscation", "quote-escape followed by comment terminator")

    # If the code already ran, only flag UNIONs missing a table so we don't block normal UNION queries
    if as_fallback:
        if has_union_injection(norm) and union_arm_has_no_table(text):
            return hit("union_based", "UNION SELECT with tableless arm")
    else:
        # If we're only doing static checks, make sure it actually looks like an attack to avoid false alarms
        if has_union_injection(norm) and has_injection_context(text):
            return hit("union_based", "UNION SELECT injection pattern")

    probe, ptype = has_injection_probe(text)
    if probe: return hit(ptype, f"injection probe pattern: {ptype}")

    # Don't trip up on Excel error codes before checking for comments
    if (stripped.startswith(("--", "/*", "#"))
            and not re.match(r"^#[A-Z]{1,10}\??$", stripped.strip())
            and not re.search(r"(?:--|/\*|#)\s*(?:SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b", stripped, re.IGNORECASE)):
        return hit("comment_obfuscation", "comment-prefix injection fragment")

    if re.match(r"^0x[0-9a-fA-F]{8,}$", stripped.replace(" ", "")):
        return hit("encoding_obfuscation", "raw hex-encoded payload")
    if re.match(r"^@[A-Za-z_]\w*$", stripped) or re.match(r"^@[A-Za-z_]\w*\s", stripped):
        return hit("blind_time", "MSSQL/MySQL session variable in payload")

    # Catch raw hex codes that are trying to hide SQL
    if re.search(r'\\x[0-9a-fA-F]{2}', text) and SQL_GATE_RE.search(norm):
        return hit("encoding_obfuscation", "literal hex escape with SQL content")

    # For SELECT TOP, make sure it looks malicious so we don't break valid SQL Server queries
    if not as_fallback and SELECT_TOP_FRAG_RE.search(norm):
        if has_injection_context(text) or re.search(r"\bunion\b", norm, re.IGNORECASE):
            return hit("blind_time", "SELECT TOP - MSSQL-specific syntax")
    if not as_fallback and re.search(r"\bunion\s+all\b", norm, re.IGNORECASE) and not re.search(r"\bselect\b", norm, re.IGNORECASE):
        return hit("union_based", "UNION ALL fragment without SELECT")

    if skip_sql_gate:
        return None
    if not SQL_GATE_RE.search(norm):
        return None

    first_tok = norm.split()
    if first_tok and first_tok[0].upper() in {"FROM", "WHERE", "LIMIT", "OFFSET", "ORDER", "INNER", "OUTER", "LEFT", "RIGHT", "JOIN"}:
        return None

    return None


def union_arm_has_no_table(sql):
    clean = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    parts = re.split(r"\bUNION\s+(?:ALL\s+)?", clean, flags=re.IGNORECASE)
    if len(parts) <= 1:
        return False
    return any(not re.search(r"\bFROM\b", arm, re.IGNORECASE) for arm in parts[1:])

# --- 9. Fragment templates ---

# Each template has a host query, the table layout, and a safe version to compare against
# pick the right template based on the context index

TEMPLATE_STRING_AUTH = (
    "SELECT id,username,role FROM users WHERE username='zzz_{p}' AND password='x'",
    {"users": {"id": "INTEGER", "username": "TEXT", "role": "TEXT", "password": "TEXT"}},
    "SELECT id,username,role FROM users WHERE username='zzz_SAFE' AND password='x'",
)

TEMPLATE_NUMERIC = (
    "SELECT id,username FROM users WHERE id={p}",
    {"users": {"id": "INTEGER", "username": "TEXT", "password": "TEXT", "role": "TEXT"}},
    "SELECT id,username FROM users WHERE id=99999999",
)

TEMPLATE_STRING_PRODUCT = (
    "SELECT name,price FROM products WHERE category='{p}'",
    {"products": {"id": "INTEGER", "name": "TEXT", "category": "TEXT", "price": "TEXT"}},
    "SELECT name,price FROM products WHERE category='__SAFE__'",
)

TEMPLATE_PAREN_STRING = (
    "SELECT id,username FROM users WHERE (username='{p}')",
    {"users": {"id": "INTEGER", "username": "TEXT", "password": "TEXT", "role": "TEXT"}},
    "SELECT id,username FROM users WHERE (username='__SAFE__')",
)

TEMPLATE_STACKED = (
    "SELECT id FROM orders WHERE status='active'; {p}",
    {"orders": {"id": "INTEGER", "status": "TEXT", "user_id": "INTEGER"}},
    "SELECT id FROM orders WHERE status='active'; SELECT 1",
)

CONTEXT_TEMPLATES = {
    0: [],
    1: [TEMPLATE_NUMERIC],
    2: [TEMPLATE_STRING_AUTH, TEMPLATE_STRING_PRODUCT],
    3: [TEMPLATE_STRING_AUTH, TEMPLATE_STRING_PRODUCT],
    4: [TEMPLATE_NUMERIC],
    5: [TEMPLATE_STRING_AUTH, TEMPLATE_STRING_PRODUCT],
    6: [TEMPLATE_PAREN_STRING],
    7: [TEMPLATE_NUMERIC],
    8: [TEMPLATE_STACKED],
    9: [TEMPLATE_STACKED],
}

NUMERIC_CONTEXTS = {1, 4, 7}
STRING_CONTEXTS = {2, 3, 5}
STACKED_CONTEXTS = {8, 9}

# --- 10. Behavioral detection ---

def detect_behavioral(sql, rows, error, baseline, input_text, mode, schema, node_set=None):
    # Check the results to see if the database acted weird. Return a hit if it did, otherwise nothing
    if node_set is None:
        node_set = set()

    def hit(exploit_type, reason):
        return SandboxResult(
            malicious=True, exploit_type=canonical(exploit_type), mode=mode,
            input_text=input_text, executed_sql=sql, rows_returned=rows,
            inferred_schema=schema, detection_reason=reason, error=error, executed=True,
        )

    # Since this is SQLite, if it complains about syntax for another database type, that's a red flag
    if error:
        err_lo = error.lower()
        for sig in CROSS_DB_ERRORS:
            if sig in err_lo:
                return hit("blind_time", f"cross-DB reference triggered error: {error[:100]}")
        
        # If it complains about multiple statements, only flag it if there are other attack signs
        if "you can only execute one statement at a time" in err_lo:
            if has_injection_context(input_text):
                return hit("stacked_queries", f"multi-statement injection confirmed: {error[:80]}")

    # If the trap secret shows up, a UNION attack successfully stole data
    if secrets_canary_in_rows(rows):
        return hit("union_based", "UNION exfiltration: secrets canary reached via UNION arm")

    # If the regular trap row shows up, an always-true condition dumped the whole table
    # Skip this check if we didn't run a baseline test to compare against
    if baseline >= 0:
        c = canary_in_rows(rows)
        if c:
            if "union" in normalize(sql).lower():
                return hit("union_based", f"UNION reached canary data: '{c[:30]}'")
            return hit("tautology", f"canary row returned by tautology condition: '{c[:30]}'")

    if baseline >= 0 and len(rows) > baseline:
        if re.search(r"\bunion\b", normalize(sql), re.IGNORECASE):
            return hit("union_based", f"UNION injection: returned {len(rows)} rows, baseline was {baseline}")
        return hit("tautology", f"tautology: WHERE clause returned {len(rows)} rows, baseline was {baseline}")

    if error and "selects to the left and right of union" in error.lower():
        return hit("union_based", f"UNION column-count mismatch: {error[:80]}")

    if error:
        for sig in INJECTION_ERRORS_COMPLETE:
            if sig in error.lower():
                return hit("blind_boolean", f"arithmetic/range error from injection: {error[:100]}")

    # If the syntax is broken and there are attack signs, flag it (strict to avoid false alarms)
    if error:
        structural_sigs = ("syntax error", "unrecognized token", "near ", "incomplete input", "no such function", "result columns")
        inject_re = re.compile(
            r"or\s+\d+\s*=\s*\d+|union\s+select|/\*|(?<!\w)--(?:\s|$)|0x[0-9a-fA-F]{4,}|declare\s+@|char\s*\(|@@\w|\bsleep\s*\(|\bwaitfor\b|\bexec\w*\s+\w",
            re.IGNORECASE,
        )
        high_confidence_re = re.compile(r"union\s+select|declare\s+@|@@\w|\bsleep\s*\(|\bwaitfor\b", re.IGNORECASE)
        if any(s in error.lower() for s in structural_sigs) and inject_re.search(input_text):
            near_match = re.search(r'near "([^"]+)": syntax error', error, re.IGNORECASE)
            if near_match:
                tok = near_match.group(1)
                if not (re.match(r"^[a-zA-Z_]\w*$", tok) or re.match(r"^[=<>!\-]+$", tok) or re.match(r"^[\d.]+$", tok)):
                    return hit("blind_boolean", f"obfuscated injection caused structural error: {error[:100]}")
            elif high_confidence_re.search(input_text):
                return hit("blind_boolean", f"structural SQL error with injection signals: {error[:100]}")

    # Catch blind boolean attacks by noticing if the number of rows changes unexpectedly
    if node_set is not None and node_set & {"Case", "If"} and baseline >= 0 and len(rows) != baseline:
        return hit("blind_boolean", f"blind boolean probe: row count changed from {baseline} to {len(rows)}")

    return None


# --- 11. Sandbox class ---

class SQLiSandbox:

    @staticmethod
    def exec_sql(conn, sql):
        try:
            return list(conn.execute(sql).fetchall()), None
        except sqlite3.Error as e:
            return [], str(e)
        except Exception as e:
            return [], f"unexpected error: {e}"

    def test(self, text, ast_profile):
        text = text.strip()

        if not ast_profile.is_valid:
            return self.static_filters_only(text)

        if ast_profile.winning_context_index == 0:
            return self.test_complete(text, ast_profile)
        elif ast_profile.winning_context_index > 0:
            return self.test_fragment(text, ast_profile)
        else:
            return self.static_filters_only(text)

    def test_batch(self, items):
        results = []
        for text, profile in items:
            results.append(self.test(text, profile))
        return results

    def static_filters_only(self, text):
        result = run_static_filters(text, mode="static_only")
        if result is not None:
            return result
        return SandboxResult(
            malicious=False, exploit_type=None, mode="static_only",
            input_text=text, executed_sql="", rows_returned=[], inferred_schema={},
            detection_reason="invalid syntax, no SQL injection patterns detected",
            executed=False,
        )

    def test_complete(self, text, profile):
        mode = "complete_query"
        blueprint = build_blueprint_from_profile(profile)
        schema = blueprint.tables

        def benign(reason, exec_rows=None, exec_error=None):
            return SandboxResult(
                malicious=False, exploit_type=None, mode=mode,
                input_text=text, executed_sql=text, rows_returned=exec_rows or [],
                inferred_schema=schema, detection_reason=reason, error=exec_error, executed=True,
            )

        conn = build_db_from_blueprint(blueprint)
        rows, error = self.exec_sql(conn, text)
        conn.close()

        if blueprint.expect_cross_db_error and error:
            err_lo = error.lower()
            for sig in CROSS_DB_ERRORS:
                if sig in err_lo:
                    return SandboxResult(
                        malicious=True, exploit_type="blind_time", mode=mode,
                        input_text=text, executed_sql=text, rows_returned=rows,
                        inferred_schema=schema, detection_reason=f"cross-DB error confirmed: {error[:100]}",
                        error=error, executed=True,
                    )

        # Catch OR-based always-true conditions that dump the trap rows. Make sure there's actually an OR to avoid false alarms
        if profile.node_set and profile.node_set & {"Or"} and rows:
            c = canary_in_rows(rows)
            if c:
                if re.search(r"\bunion\b", normalize(text), re.IGNORECASE):
                    return SandboxResult(
                        malicious=True, exploit_type="union_based", mode=mode,
                        input_text=text, executed_sql=text, rows_returned=rows, inferred_schema=schema,
                        detection_reason=f"OR-tautology + UNION exposed canary data: '{c[:30]}'",
                        error=error, executed=True,
                    )
                return SandboxResult(
                    malicious=True, exploit_type="tautology", mode=mode,
                    input_text=text, executed_sql=text, rows_returned=rows, inferred_schema=schema,
                    detection_reason=f"tautology: canary row returned by OR condition: '{c[:30]}'",
                    error=error, executed=True,
                )

        behavioral = detect_behavioral(
            sql=text, rows=rows, error=error, baseline=-1,
            input_text=text, mode=mode, schema=schema, node_set=profile.node_set,
        )
        if behavioral is not None:
            return behavioral

        static_result = run_static_filters(text, mode=mode, skip_sql_gate=True, as_fallback=True)
        if static_result is not None:
            return SandboxResult(
                malicious=static_result.malicious, exploit_type=static_result.exploit_type, mode=mode,
                input_text=text, executed_sql=text, rows_returned=rows, inferred_schema=schema,
                detection_reason=static_result.detection_reason, error=error, executed=True,
            )

        return benign("no exploit detected", exec_rows=rows, exec_error=error)

    # Test fragment of a query by plugging it into different templates 
    def test_fragment(self, text, profile):
        mode = "payload_fragment"
        norm = normalize(text)
        stripped = text.strip()

        def hit(exploit_type, reason):
            return SandboxResult(
                malicious=True, exploit_type=canonical(exploit_type), mode=mode,
                input_text=text, executed_sql=text, rows_returned=[], inferred_schema={},
                detection_reason=reason, executed=False,
            )

        # Stop early for attack signs that SQLite won't naturally throw an error for
        if (stripped.startswith(("--", "/*", "#"))
                and not re.match(r"^#[A-Z]{1,10}\??$", stripped.strip())
                and not re.search(
                    r"(?:--|/\*|#)\s*(?:SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b", stripped, re.IGNORECASE
                )):
            return hit("comment_obfuscation", "comment-prefix injection fragment")
        if re.match(r"^@[A-Za-z_]\w*($|\s)", stripped):
            return hit("blind_time", "MSSQL/MySQL session variable fragment")
        if re.match(r"^0x[0-9a-fA-F]{8,}$", stripped.replace(" ", "")):
            return hit("encoding_obfuscation", "hex-encoded fragment payload")
        if re.search(r"\bunion\s+all\b", norm, re.IGNORECASE) and not re.search(r"\bselect\b", norm, re.IGNORECASE):
            return hit("union_based", "UNION ALL fragment without SELECT")

        ctx_idx = profile.winning_context_index
        node_set = profile.node_set or set()
        templates = list(CONTEXT_TEMPLATES.get(ctx_idx, []))

        # Pick the template that matches the number of columns best, so UNION tests line up
        if UNION_NODES & node_set and profile.select_arm_widths:
            arm_width = profile.select_arm_widths[0]
            if arm_width > 2:
                if TEMPLATE_STRING_AUTH not in templates:
                    templates = [TEMPLATE_STRING_AUTH] + templates
            else:
                if TEMPLATE_NUMERIC not in templates:
                    templates = [TEMPLATE_NUMERIC] + templates

        if has_stacked_query(text):
            templates = [TEMPLATE_STACKED] + [t for t in templates if t != TEMPLATE_STACKED]

        if has_cross_db(norm) and not templates:
            templates = [TEMPLATE_STRING_AUTH, TEMPLATE_NUMERIC]

        if not templates:
            templates = [TEMPLATE_STRING_AUTH, TEMPLATE_NUMERIC]

        last_clean = None
        templates_ran = False
        last_rows = []
        last_error = None
        last_injected = ""
        last_blueprint_tables = {}

        for tmpl_sql, tmpl_schema, safe_sql in templates:
            injected = tmpl_sql.replace("{p}", text)
            blueprint = build_blueprint_from_template(tmpl_schema, node_set, profile.select_arm_widths)

            conn = build_db_from_blueprint(blueprint)
            baseline = len(self.exec_sql(conn, safe_sql)[0])
            rows, error = self.exec_sql(conn, injected)
            conn.close()

            templates_ran = True
            last_rows = rows
            last_error = error
            last_injected = injected
            last_blueprint_tables = blueprint.tables

            behavioral = detect_behavioral(
                sql=injected, rows=rows, error=error, baseline=baseline,
                input_text=text, mode=mode, schema=blueprint.tables, node_set=node_set,
            )
            if behavioral is not None:
                return behavioral

            last_clean = SandboxResult(
                malicious=False, exploit_type=None, mode=mode,
                input_text=text, executed_sql=injected, rows_returned=rows,
                inferred_schema=blueprint.tables, detection_reason="no exploit detected in template execution",
                error=error, executed=True,
            )

        static_result = run_static_filters(text, mode=mode, as_fallback=True)
        if static_result is not None:
            if templates_ran:
                return SandboxResult(
                    malicious=static_result.malicious, exploit_type=static_result.exploit_type, mode=mode,
                    input_text=text, executed_sql=last_injected, rows_returned=last_rows,
                    inferred_schema=last_blueprint_tables, detection_reason=static_result.detection_reason,
                    error=last_error, executed=True,
                )
            return static_result

        return last_clean or SandboxResult(
            malicious=False, exploit_type=None, mode=mode,
            input_text=text, executed_sql="", rows_returned=[], inferred_schema={},
            detection_reason="no exploit detected across all templates", executed=True,
        )