from src.optimizer import build_optimized_squad
from src.transfer_logic import inspect_transfer_logic

# Use the optimizer squad as a temporary test current squad
optimized_squad = build_optimized_squad(use_cache=True, verbose=False)

current_squad_names = optimized_squad["name"].tolist()

# Optional: define a simple starting XI from the optimized squad
starting_names = optimized_squad.sort_values("predicted_points", ascending=False).head(11)["name"].tolist()

outputs = inspect_transfer_logic(
    current_squad_names=current_squad_names,
    money_in_bank=0.0,
    starting_names=starting_names,
    use_cache=True,
    verbose=True,
)

print("\nTransfer logic test complete.")