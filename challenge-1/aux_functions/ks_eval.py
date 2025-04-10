import numpy as np
import matplotlib.pyplot as plt

k=20 # number of snapshots to test for short time and long time
modes = 20 # Need modes strictly less than m/2

# SCORING FOR SHORT-TIME AND LONG-TIME FORECASTS
def scoring_ks(truth, prediction, k, modes):
    '''produce long-time and short-time error scores.'''
    [m,n]=truth.shape
    Est = np.linalg.norm(truth[:,0:k]-prediction[:,0:k],2)/np.linalg.norm(truth[:,0:k],2)

    m2 = 2*modes+1
    Pt = np.empty((m2,0))
    Pp = np.empty((m2,0))

    # LONG TIME:  Compute least-square fit to power spectra
    for j in range(1, k+1):
        P_truth = np.multiply(np.abs(np.fft.fft(truth[:, n-j])), np.abs(np.fft.fft(truth[:, n-j])))
        P_prediction = np.multiply(np.abs(np.fft.fft(prediction[:, n-j])), np.abs(np.fft.fft(prediction[:, n-j])))
        Pt3 = np.fft.fftshift(P_truth)
        Pp3 = np.fft.fftshift(P_prediction)
        Ptnew = Pt3[int(m/2)-modes:int(m/2)+modes+1]
        Ppnew = Pp3[int(m/2)-modes:int(m/2)+modes+1]  # Fixed the variable name
    
        Pt = np.column_stack((Pt, np.log(Ptnew)))
        Pp = np.column_stack((Pp, np.log(Ppnew)))
    
    Elt = np.linalg.norm(Pt-Pp,2)/np.linalg.norm(Pt,2)
    
    E1 = 100*(1-Est)
    E2 = 100*(1-Elt)

    if np.isnan(E1):
        E1 = -np.inf
    if np.isnan(E2):
        E2 = -np.inf
    
    
    return E1, E2


if __name__ == "__main__":
    import os

    DATA_FOLDER = "data"
    TEAM_FOLDER = "team_entries/team0"
    TRUTH_FILE = os.path.join(DATA_FOLDER, "ks_truth.npy")
    PREDICTION_FILE = os.path.join(TEAM_FOLDER, "ks_prediction.npy")

    truth = np.load(TRUTH_FILE)
    prediction = np.load(PREDICTION_FILE)
    E1, E2 = scoring_ks(truth, prediction, k, modes)
    print(E1)
    print(E2)