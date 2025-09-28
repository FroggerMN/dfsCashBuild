import pandas as pd


def load_split_dk_results(week:str):
    """
    Load and split DraftKings contest results into lineups and player pool.

    Args:
        week (str): The week identifier for the data folder.
        file_name (str): The name of the CSV file containing contest standings.
    """

    # Load the CSV file
    file_path = f'data/{week}/contest-standings.csv'
    df = pd.read_csv(file_path)

    # Identify the index of 'Unnamed: 6' (or similar placeholder)
    # This acts as the split point
    split_col = 'Unnamed: 6'

    if split_col not in df.columns:
        raise ValueError(f"Column '{split_col}' not found. Please check the actual name.")

    split_idx = df.columns.get_loc(split_col)

    # Split the DataFrame
    entries_df = df.iloc[:, :split_idx]  # Everything before 'Unnamed: 6'
    # Filter rows based on column: 'Lineup'
    entries_df = entries_df[entries_df['Lineup'].notna()]
    playerpool_df = df.iloc[:, split_idx:]  # From 'Unnamed: 6' onward


    # Clean up playerpool: drop the NaN column if it's just placeholder
    # Rename the first column to something meaningful if needed
    playerpool_df = playerpool_df.dropna(axis=1, how='all')  # Remove all-NaN columns
    return playerpool_df, entries_df
def load_player_data(week:str, playerpool_df):
    """
    Load player data for the given week.

    Args:
        week (str): The week identifier for the data folder.
    """
    player_data_path = f'data/{week}/dk_player_data.csv'
    player_data = pd.read_csv(player_data_path)
    player_results_df = playerpool_df.merge(player_data, left_on='Player', right_on='name', how='left')
    # Strip ALL leading-and-trailing whitespace (tabs, new-lines, spaces, etc.)
    player_results_df['Player'] = player_results_df['Player'].str.strip()
    return player_results_df

def parse_lineup_string(lineup_str):
    """
    Parse the lineup string to extract individual player names
    Example: "DST Ravens FLEX George Pickens QB Jared Goff RB Christian McCaffrey..."
    """
    if pd.isna(lineup_str):
        return []
    
    # Split the lineup string and look for player names
    # Positions are typically: DST, QB, RB, WR, TE, FLEX
    positions = ['DST', 'QB', 'RB', 'WR', 'TE', 'FLEX']
    
    players = []
    parts = lineup_str.split()
    
    i = 0
    while i < len(parts):
        if parts[i] in positions:
            # Found a position, next part(s) should be player name
            i += 1
            player_name_parts = []
            
            # Collect name parts until we hit another position or end
            while i < len(parts) and parts[i] not in positions:
                player_name_parts.append(parts[i])
                i += 1
            
            if player_name_parts:
                player_name = ' '.join(player_name_parts)
                players.append(player_name)
        else:
            i += 1
    
    return players

def calculate_player_score(player_name, player_results_df,
                           projown_wt=0.5, projpts_wt=0.5):
    """
    Return weighted score for one player.
    """
    # locate the player (case-insensitive)
    mask = player_results_df['Player'].str.lower() == player_name.lower()
    player_match = player_results_df[mask]

    if player_match.empty:
        # player not found – log & give zero contribution
        print(f"⚠️  '{player_name}' not found in player_results")
        return 0.0

    # take the first matching row
    player_row = player_match.iloc[0]

    projown = pd.to_numeric(player_row.get('projown', 0), errors='coerce')
    projpts = pd.to_numeric(player_row.get('projpts', 0), errors='coerce')

    projown = 0 if pd.isna(projown) else projown
    projpts = 0 if pd.isna(projpts) else projpts

    return projown_wt * projown + projpts_wt * projpts

def calculate_lineup_rating(entries_df, player_results_df, projown_wt=0.5, projpts_wt=0.5):
    """
    Calculate lineup_rating for each entry
    """
    lineup_ratings = []
    
    for idx, row in entries_df.iterrows():
        lineup_str = row['Lineup']
        players_in_lineup = parse_lineup_string(lineup_str)
        
        total_rating = 0
        player_count = 0
        
        for player in players_in_lineup:
            player_score = calculate_player_score(
                player, player_results_df, projown_wt, projpts_wt
            )
            total_rating += player_score
            player_count += 1
        
        lineup_ratings.append({
            'EntryId': row['EntryId'],
            'EntryName': row['EntryName'],
            'Rank': row['Rank'],
            'Points': row['Points'],
            'lineup_rating': total_rating,
            'players_found': player_count
        })
 
    
    return pd.DataFrame(lineup_ratings)

def engineer_lineup_features(lineup_str, player_results_df):
    """
    Calculate all required features for a given lineup.
    
    Args:
        lineup_str: String representation of the lineup
        player_results_df: DataFrame containing player projections
        
    Returns:
        dict: Dictionary of calculated features
    """
    # Parse lineup into player list if not already parsed
    if isinstance(lineup_str, str):
           
        players = parse_lineup_string(lineup_str)
    else:
        players = lineup_str
        
    # Initialize features dictionary
    features = {
        'sum_projpts': 0,
        'sum_projown': 0,
        'avg_floor': 0,
        'num_value_players': 0,
        'num_chalk_players': 0,
        'qb_wr_stacks': 0,
        'num_stacks': 0
    }
    
    # Track player data for stack calculation
    player_data = []
    
    # Calculate per-player features
    for player_name in players:
        # Find player in results df (case insensitive)
        mask = player_results_df['Player'].str.lower() == player_name.lower()
        if mask.any():
            player_row = player_results_df.loc[mask].iloc[0]
            
            # Basic sums
            features['sum_projpts'] += float(player_row.get('projpts', 0))
            proj_own = float(player_row.get('projown', 0))
            features['sum_projown'] += proj_own
            
            # Value and chalk calculations
            if proj_own < 20:
                features['num_value_players'] += 1
            if proj_own > 60:
                features['num_chalk_players'] += 1
                
            # Floor calculation
            features['avg_floor'] += float(player_row.get('avgpointspergame', 0))
            
            # Store player data for stack calculation
            player_data.append({
                'name': player_name,
                'position': player_row.get('rosterposition', ''),
                'team': player_row.get('teamabbrev', '')
            })
    
    # Calculate average floor
    if players:
        features['avg_floor'] /= len(players)
 

def add_lineup_features(entries_df, player_results_df):
    """
    Add all lineup features to entries dataframe.
    
    Args:
        entries_df: DataFrame with lineup entries
        player_results_df: DataFrame with player projections
        
    Returns:
        DataFrame: entries_df with added features
    """
    # Apply feature engineering function to each row
    features_list = entries_df['Lineup'].apply(
        lambda x: engineer_lineup_features(x, player_results_df)
    ).tolist()
    
    # Convert list of dicts to DataFrame and join with entries_df
    features_df = pd.DataFrame(features_list, index=entries_df.index)
    
    # Join features with original dataframe
    return pd.concat([entries_df, features_df], axis=1)

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
def calculate_baseline_rating(df):
    """
    Calculate baseline rating as 0.5*projown + 0.5*projpts.
    
    Args:
        df: DataFrame with sum_projown and sum_projpts columns
        
    Returns:
        Series: Baseline ratings
    """
    # Normalize first to ensure equal weighting
    norm_projpts = (df['sum_projpts'] - df['sum_projpts'].mean()) / df['sum_projpts'].std()
    norm_projown = (df['sum_projown'] - df['sum_projown'].mean()) / df['sum_projown'].std()
    
    return 0.5 * norm_projown + 0.5 * norm_projpts

def export_training_data(entries_df, feature_columns, target_columns=None, filename='training_data.csv'):
    """
    Export training data with features and targets.
    
    Args:
        entries_df: DataFrame with features and targets
        feature_columns: List of feature column names
        target_columns: List of target column names (default: None)
        filename: Output CSV filename (default: 'training_data.csv')
        
    Returns:
        DataFrame: The exported data
    """
    # Select columns to export
    cols_to_export = feature_columns.copy()
    if target_columns:
        cols_to_export.extend(target_columns)
    
    # Select only existing columns
    existing_cols = [col for col in cols_to_export if col in entries_df.columns]
    export_df = entries_df[existing_cols].copy()
    
    # Export to CSV
    export_df.to_csv(filename, index=False)
    
    return export_df






