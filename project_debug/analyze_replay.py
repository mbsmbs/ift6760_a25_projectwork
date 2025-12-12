import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1) Point this to the HYDRA run dir you used ---
# Example from your last run:
# /Users/byungsukmin/.../logs/hybrid_extended/debug_reward_03/847e74f9
RUN_DIR = "/Users/byungsukmin/Desktop/udem/INF/IFT6760/Project/ift6760_a25_projectwork/logs/hybrid_extended/debug_reward_03/847e74f9"

# If you saved the replay buffer as a CSV manually, specify it here.
# Otherwise, we will try to auto-detect a *.csv inside RUN_DIR.
REPLAY_CSV = None  # e.g. os.path.join(RUN_DIR, "replay_buffer.csv")

COL_SCORE = "energies"  # score column in gfn_samples.csv


def find_replay_csv(run_dir):
    if REPLAY_CSV is not None and os.path.isfile(REPLAY_CSV):
        return REPLAY_CSV

    # Search for any csv in the run dir
    candidates = glob.glob(os.path.join(run_dir, "*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV file found in {run_dir}. "
                                "You may need to export the replay buffer to CSV first.")
    # Pick the first one for now
    return candidates[0]


def main():
    csv_path = find_replay_csv(RUN_DIR)
    print(f"Loading replay buffer from: {csv_path}")
    df = pd.read_csv(csv_path)

    print("\nColumns:", list(df.columns))
    print("\nHead:\n", df.head())

    # --- Stats for normalized energy / score ---
    if COL_SCORE in df.columns:
        vals = df[COL_SCORE].values
        print(f"\n=== {COL_SCORE.upper()} stats ===")
        print(f"min:  {vals.min():.6g}")
        print(f"max:  {vals.max():.6g}")
        print(f"mean: {vals.mean():.6g}")
        print(f"std:  {vals.std():.6g}")
    else:
        print(f"[WARN] Column '{COL_SCORE}' not found in file.")

    # --- Histogram ---
    if COL_SCORE in df.columns:
        plt.figure()
        plt.hist(df[COL_SCORE].values, bins=50)
        plt.xlabel(COL_SCORE)
        plt.ylabel("count")
        plt.title("Test samples normalized energies")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
