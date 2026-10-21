import numpy as np
from scipy.signal import welch


def extract_features(X, sampling_rate):

    all_features = []

    for trial in X:

        features = []

        # -----------------------------------------
        # Time-domain features
        # -----------------------------------------

        mean = np.mean(
            trial,
            axis=1
        )

        std = np.std(
            trial,
            axis=1
        )

        variance = np.var(
            trial,
            axis=1
        )

        maximum = np.max(
            trial,
            axis=1
        )

        minimum = np.min(
            trial,
            axis=1
        )

        features.extend(mean)
        features.extend(std)
        features.extend(variance)
        features.extend(maximum)
        features.extend(minimum)

        # -----------------------------------------
        # Frequency-domain features
        # -----------------------------------------

        frequencies, power = welch(
            trial,
            fs=sampling_rate,
            axis=1
        )

        bands = [
            (8, 12),
            (12, 16),
            (16, 20),
            (20, 30)
        ]

        for low, high in bands:

            mask = (
                (frequencies >= low) &
                (frequencies < high)
            )

            band_power = np.mean(
                power[:, mask],
                axis=1
            )

            # Log transform makes power values
            # easier for the classifier to use
            log_band_power = np.log(
                band_power + 1e-10
            )

            features.extend(log_band_power)

        all_features.append(features)

    return np.array(all_features)