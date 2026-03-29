from src.api import get_base_fpl_tables

tables = get_base_fpl_tables()

for name, df in tables.items():
    print(f"\n{name.upper()}")
    print(df.shape)
    print(df.head())