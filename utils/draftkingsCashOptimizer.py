import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus
import logging
from typing import List, Optional, Dict, Any


def column_str_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all column names to lowercase and strip whitespace."""
    return df.rename(columns=lambda x: x.strip().lower())


def fetch_player_data(
    week: str="week01",
    dk_salaries_file: str = "DKSalaries.csv",
    dk_projections_file: str = "fantasy_footballers-nfl-dk-Main-projections.csv",
) -> pd.DataFrame:
    """
    Load and merge DraftKings salary and projection data.
    Args:
        dk_salaries_path: Path to DK salaries CSV.
        projections_path: Path to projections CSV.
    Returns:
        Merged DataFrame with standardized columns.
    """
    try:
        dk_df = column_str_standardize(pd.read_csv(f"data/{week}/{dk_salaries_file}"))
        proj_df = column_str_standardize(pd.read_csv(f"data/{week}/{dk_projections_file}"))
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise
    try:
        merged = pd.merge(dk_df, proj_df, on=["id", "position"], how="inner")
        merged.drop(columns=["name_y"], inplace=True)
        merged.rename(columns={"name_x": "name"}, inplace=True)
        desired_columns = [
            "id",
            "name",
            "position",
            "roster position",
            "salary",
            "game info",
            "team",
            "avgpointspergame",
            "projpts",
            "projown",
        ]
        merged.to_csv(f"data/{week}/dk_player_data.csv", index=False)
        return merged[desired_columns]
    except Exception as e:
        logging.error(f"Error merging data: {e}")
        raise

def normalize_features(df, feature_columns):
    """
    Normalize feature columns to have mean=0 and std=1.
    
    Args:
        df: DataFrame with features
        feature_columns: List of columns to normalize
        
    Returns:
        DataFrame: df with normalized features
    """
    df_normalized = df.copy()
    
    for col in feature_columns:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:  # Avoid division by zero
                df_normalized[col] = (df[col] - mean) / std
    
    return df_normalized

def generate_top_lineups(
    player_data: pd.DataFrame,
    num_lineups: int = 5,
    locked_players: Optional[List[Any]] = None,
    projown_wt: float = 0.1,
    projpts_wt: float = 0.9,
) -> List[Dict[str, Any]]:
    """
    Generate the top N optimal lineups, optionally locking specific players.
    Args:
        player_data: DataFrame of all available players.
        num_lineups: Number of unique lineups to generate.
        locked_players: List of player 'id' values to lock (optional).
    Returns:
        List of dicts with 'score' and 'lineup' DataFrame.
    """
    logger = logging.getLogger("LineupOptimizer")
    all_lineups = []
    found_player_ids = []

    # Validate locked players
    if locked_players:
        logger.info("--- Validating Locked Players ---")
        all_player_ids = set(player_data["id"])
        for pid in locked_players:
            if pid not in all_player_ids:
                raise ValueError(
                    f"Locked player with ID '{pid}' not found in player data."
                )
        locked_df = player_data[player_data["id"].isin(locked_players)]
        locked_pos_counts = locked_df["position"].value_counts()
        logger.info(f"Locked Player Counts by Position:\n{locked_pos_counts}")
        if locked_pos_counts.get("QB", 0) > 1:
            logger.warning("Locking >1 QB will result in an infeasible lineup.")
        if locked_pos_counts.get("RB", 0) > 3:
            logger.warning("Locking >3 RBs will result in an infeasible lineup.")
        if locked_pos_counts.get("WR", 0) > 4:
            logger.warning("Locking >4 WRs will result in an infeasible lineup.")
        if locked_pos_counts.get("TE", 0) > 1:
            logger.warning("Locking >1 TEs will result in an infeasible lineup.")
        if locked_pos_counts.get("DST", 0) > 1:
            logger.warning("Locking >1 DEF will result in an infeasible lineup.")

    for i in range(num_lineups):
        logger.info(f"--- Finding Lineup #{i + 1} ---")
        model = LpProblem(f"Fantasy_Lineup_{i + 1}", LpMaximize)
        player_vars = {
            row["id"]: LpVariable(f"player_{row['id']}", cat="Binary")
            for _, row in player_data.iterrows()
        }

        model += (
            lpSum(
                player_vars[row["id"]] * (0.1 * row["projpts"] + 0.9 * row["projown"])
                for _, row in player_data.iterrows()
            ),
            "Maximize_Figure_of_Merit",
        )

        model += lpSum(player_vars.values()) == 9, "Total_9_Players"
        model += (
            lpSum(
                player_vars[row["id"]] * row["salary"]
                for _, row in player_data.iterrows()
            )
            <= 50000,
            "Salary_Cap",
        )

        positions = ["QB", "RB", "WR", "TE", "DST"]
        for pos in positions:
            pos_players = player_data[player_data["position"] == pos]
            pos_count = lpSum(
                player_vars[row["id"]] for _, row in pos_players.iterrows()
            )
            if pos == "QB":
                model += pos_count <= 1, f"Max_1_QB_{i}"
            elif pos == "RB":
                model += 2 <= pos_count <= 3, f"RB_Range_{i}"
            elif pos == "WR":
                model += 3 <= pos_count <= 4, f"WR_Range_{i}"
            elif pos == "TE":
                model += pos_count <= 1, f"Max_1_TE_{i}"
            elif pos == "DST":
                model += pos_count == 1, f"Exactly_1_DEF_{i}"

        # Add constraints for locked players
        if locked_players:
            for player_id in locked_players:
                model += player_vars[player_id] == 1, f"Lock_Player_{player_id}"

        # Exclude previously found lineups
        for prev_ids in found_player_ids:
            model += (
                lpSum(player_vars[pid] for pid in prev_ids) <= 8,
                f"Exclude_Lineup_{found_player_ids.index(prev_ids)}",
            )

        status = model.solve()

        if LpStatus[status] == "Optimal" and model.objective is not None:
            score = model.objective.value()
            selected_ids = [
                pid for pid, var in player_vars.items() if var.varValue == 1
            ]
            found_player_ids.append(selected_ids)
            lineup_df = player_data[player_data["id"].isin(selected_ids)].copy()
            all_lineups.append({"score": score, "lineup": lineup_df})
            logger.info(f"Status: Optimal. Score: {score:.2f}\n")
        else:
            logger.warning(
                f"Could not find an optimal solution for lineup #{i + 1}. Stopping."
            )
            break

    return all_lineups
