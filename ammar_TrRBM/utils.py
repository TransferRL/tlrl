from sklearn.preprocessing import StandardScaler

def standardize_samples(samples):
    scaler = StandardScaler()
    scaler.fit(samples)
    return scaler, scaler.transform(samples)