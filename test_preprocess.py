from src.preprocess import inspect_clean_history_table

df = inspect_clean_history_table(use_cache=True)
print("\nColumns:")
print(df.columns.tolist())