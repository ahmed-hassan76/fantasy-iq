from src.gw1_builder import inspect_gw1_builder

outputs = inspect_gw1_builder(
    use_cache=True,
    include_unmatched=True,
    unmatched_penalty=0.85,
    season_folder=None,   # auto-infer previous season, e.g. 2024-25
    verbose=True,
)

print("\nGW1 hybrid builder test complete.")