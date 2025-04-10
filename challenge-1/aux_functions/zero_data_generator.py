""" This file is auxiliary and allos the generation of random and zero prediction placeholder date for both systems"""

import numpy as np
import os

def generate_random_data():
    """Generate random data for both KS and Lorenz systems"""
    # KS data shape: (100, 2048)
    ks_data = np.random.randn(100, 2048)
    
    # Lorenz data shape: (5000, 3)
    lorenz_data = np.random.randn(5000, 3)
    
    return ks_data, lorenz_data


def generate_zero_data():
    """Generate zero data for both KS and Lorenz systems"""
    # KS data shape: (100, 2048)
    ks_data = np.zeros([100, 2048])
    
    # Lorenz data shape: (5000, 3)
    lorenz_data = np.zeros([5000, 3])
    
    return ks_data, lorenz_data

def save_data(team_number):
    """Save random/zero predictions for a team"""
    # Generate random/zero data
    ks_pred, lorenz_pred = generate_zero_data()
    
    # Create team folder if it doesn't exist
    team_folder = f"team_entries/team{team_number}"
    os.makedirs(team_folder, exist_ok=True)
    
    # Save predictions
    ks_file = os.path.join(team_folder, "ks_prediction.npy")
    lorenz_file = os.path.join(team_folder, "lorenz_prediction.npy")
    
    np.save(ks_file, ks_pred)
    np.save(lorenz_file, lorenz_pred)
    
    # Create team name file if it doesn't exist
    teamname_file = os.path.join(team_folder, "teamname.txt")
    if not os.path.exists(teamname_file):
        with open(teamname_file, "w") as f:
            f.write(f"Team {team_number}")
    
    print(f"Generated and saved random predictions for {team_folder}")
    print(f"KS prediction shape: {ks_pred.shape}")
    print(f"Lorenz prediction shape: {lorenz_pred.shape}")

if __name__ == "__main__":
    # Generate data for specified team
    save_data(3)