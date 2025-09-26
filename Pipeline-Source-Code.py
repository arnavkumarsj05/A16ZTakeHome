import dlt
from pyspark.sql import functions as F, types as T
from pyspark.sql.window import Window

# ===============================
# Expected columns/dtypes
# ===============================

# Columns we expect in the FINAL table
EXPECTED_FINAL_COLS = [
    "id", "Commitment", "VehicleType", "LimitedPartner", "FundName",
    "Organization_name_hashed", "RelationshipType", "AccountType",
    "BiteLow", "BiteHigh", "AccountLead", "HQCityStateCountry",
    # helper flags used by expectations 
    "__has_account", "__within_3sigma_Commitment", "__within_3sigma_BiteLow", "__within_3sigma_BiteHigh",
    "__schema_ok__"
]

# Types we enforce
EXPECTED_TYPES = {
    "Commitment": "double",
    "BiteLow":    "double",
    "BiteHigh":   "double",
}

# String columns to clean
STRING_COLS_TO_CLEAN = [
    "Organization_name_hashed", "RelationshipType", "AccountType",
    "AccountLead", "HQCityStateCountry", "VehicleType", "FundName"
]

# Numeric columns
NUMERIC_COLS = ["Commitment", "BiteLow", "BiteHigh"]

# BiteLow/High are in millions -> convert to raw × 1,000,000. Commitment already raw.
MILLION = 1_000_000.0


# ==========
# Sources
# ==========
@dlt.view
def src_accounts():
    return spark.table("workspace.default.accounts")

@dlt.view
def src_commitments():
    return spark.table("workspace.default.commitments")


# ==============================
# Join (keep c.id, drop a.id)
# ==============================
@dlt.view
def joined():
    a = dlt.read("src_accounts").alias("a")
    c = dlt.read("src_commitments").alias("c")

    j = c.join(a, F.col("c.LimitedPartner") == F.col("a.id"), "full")

    # Keep commitments.*, plus all account columns EXCEPT accounts.id
    df = j.select(
        F.col("c.*"),
        F.col("a.Organization_name_hashed"),
        F.col("a.RelationshipType"),
        F.col("a.AccountType"),
        F.col("a.BiteLow"),
        F.col("a.BiteHigh"),
        F.col("a.AccountLead"),
        F.col("a.HQCityStateCountry"),
        # helper flag used for orphan reporting expectations
        F.col("a.id").isNotNull().alias("__has_account"),
    )
    return df


# ==============================
# Cleaning
# ==============================
@dlt.view
def cleaned():
    df = dlt.read("joined")

    # ASCII-only for strings
    def ascii_only(col):
        return F.regexp_replace(F.col(col).cast("string"), r"[^\x00-\x7F]", "")

    for col in STRING_COLS_TO_CLEAN:
        if col in df.columns:
            df = df.withColumn(col, ascii_only(col))

    # NULL -> 'unknown' for strings
    for col in STRING_COLS_TO_CLEAN:
        if col in df.columns:
            df = df.withColumn(col, F.coalesce(F.col(col), F.lit("unknown")))

    # Cast numerics
    for col in NUMERIC_COLS:
        if col in df.columns:
            df = df.withColumn(col, F.col(col).cast("double"))

    # Units: BiteLow/High millions -> raw
    if "BiteLow" in df.columns:
        df = df.withColumn("BiteLow", F.col("BiteLow") * F.lit(MILLION))
    if "BiteHigh" in df.columns:
        df = df.withColumn("BiteHigh", F.col("BiteHigh") * F.lit(MILLION))

    return df


# ==========================================
# Schema flag & 3-sigma outlier flags
# ==========================================
def append_schema_ok(df):
    # Enforce that all expected business columns exist with expected types (allow extras)
    field_type = {f.name: f.dataType.simpleString() for f in df.schema.fields}

    names_ok = all(c in field_type for c in [
        "id","Commitment","VehicleType","LimitedPartner","FundName",
        "Organization_name_hashed","RelationshipType","AccountType",
        "BiteLow","BiteHigh","AccountLead","HQCityStateCountry"
    ])

    types_ok = all(field_type.get(c) == t for c, t in EXPECTED_TYPES.items())

    return df.withColumn("__schema_ok__", F.lit(bool(names_ok and types_ok)))


def append_3sigma_flags(df):
    # Global mean/std (partition all) for each numeric col
    w = Window.partitionBy(F.lit(1))
    for col in [c for c in NUMERIC_COLS if c in df.columns]:
        mu = F.avg(F.col(col)).over(w)
        sd = F.stddev_samp(F.col(col)).over(w)
        within = (sd.isNull()) | (sd == 0) | F.col(col).isNull() | (F.abs((F.col(col) - mu) / sd) <= F.lit(3.0))
        df = df.withColumn(f"__within_3sigma_{col}", within)
    return df


# ======================================
# Checked table + Expectations
# ======================================
@dlt.table(
    name="reporting_table",
    comment="Table with strict failing checks and report-only expectations"
)

# --- FAIL THE PIPELINE expectations ---
@dlt.expect_or_fail("schema_ok", "__schema_ok__")
@dlt.expect_or_fail("no_negatives",
    "(Commitment >= 0 OR Commitment IS NULL) AND " +
    "(BiteLow >= 0 OR BiteLow IS NULL) AND " +
    "(BiteHigh >= 0 OR BiteHigh IS NULL)"
)
@dlt.expect_or_fail("bite_bounds_ok",
    "BiteLow IS NULL OR BiteHigh IS NULL OR BiteLow <= BiteHigh"
)

# --- REPORT-ONLY expectations ---
# Nulls
@dlt.expect("commitment_not_null", "Commitment IS NOT NULL")
@dlt.expect("bitelow_not_null",    "BiteLow IS NOT NULL")
@dlt.expect("bitehigh_not_null",   "BiteHigh IS NOT NULL")

# Orphaned LPs: commitment row with no matching account
@dlt.expect("no_orphan_lps", "__has_account OR LimitedPartner IS NULL")

# Extreme outliers (3-sigma) — counts only
@dlt.expect("commitment_within_3sigma", "__within_3sigma_Commitment")
@dlt.expect("bitelow_within_3sigma",    "__within_3sigma_BiteLow")
@dlt.expect("bitehigh_within_3sigma",   "__within_3sigma_BiteHigh")
def reporting_table():
    df = dlt.read("cleaned")
    df = append_3sigma_flags(df)
    df = append_schema_ok(df)
    return df


# ===============================
# Final table, drops helper boolean columns
# ===============================
BUSINESS_COLS = [
    "id", "Commitment", "VehicleType", "LimitedPartner", "FundName",
    "Organization_name_hashed", "RelationshipType", "AccountType",
    "BiteLow", "BiteHigh", "AccountLead", "HQCityStateCountry",
]

@dlt.table(
    name="prod_accounts_commitments_joined",
    comment="Final production-ready table."
)
def prod_accounts_commitments_joined():
    return dlt.read("reporting_table").select(*BUSINESS_COLS)
