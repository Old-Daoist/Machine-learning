"""
Assignment IX: Numerical Integration on National Water Data Portal data.
Dataset: Temperature (Telemetry Hourly), CWPRS, Maharashtra (2021-2025 CSV).
Note: Inspection of the supplied file shows records mostly every 5 minutes, not every hour.
The script selects the longest contiguous 5-minute run, then takes 289 samples (288 intervals = 24 h),
so Simpson 1/3 (even n) and Simpson 3/8 (n divisible by 3) are both valid.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_CSV = "temprature_tel_hr_cwprs_mh_2021_2025.csv"
TIME_COL = "Data Acquisition Time"
TEMP_COL = "Air Temperature Telemetry Hourly (ºC)"

def trapezoidal(y, h):
    return h * (0.5*y[0] + y[1:-1].sum() + 0.5*y[-1])

def simpson_13(y, h):
    n = len(y)-1
    if n % 2 != 0:
        raise ValueError("Simpson 1/3 requires an even number of subintervals.")
    return h/3 * (y[0] + y[-1] + 4*y[1:-1:2].sum() + 2*y[2:-1:2].sum())

def simpson_38(y, h):
    n = len(y)-1
    if n % 3 != 0:
        raise ValueError("Simpson 3/8 requires subinterval count divisible by 3.")
    weights = np.ones(n+1)
    for i in range(1, n):
        weights[i] = 2 if i % 3 == 0 else 3
    return 3*h/8 * np.dot(weights, y)

def main():
    df = pd.read_csv(INPUT_CSV)
    df["timestamp"] = pd.to_datetime(df[TIME_COL], format="%d-%m-%Y %H:%M", errors="coerce")
    df["temperature_C"] = pd.to_numeric(df[TEMP_COL], errors="coerce")
    df = df.dropna(subset=["timestamp", "temperature_C"]).sort_values("timestamp")
    df = df.drop_duplicates("timestamp").reset_index(drop=True)

    # Split where timestamps are not exactly 5 minutes apart.
    groups = (df["timestamp"].diff() != pd.Timedelta(minutes=5)).cumsum()
    sizes = df.groupby(groups).size()
    longest_id = sizes.idxmax()
    segment = df.loc[groups.eq(longest_id)].iloc[:289].copy()
    if len(segment) < 289:
        raise RuntimeError("No contiguous 24-hour 5-minute segment with 289 samples was found.")

    y = segment["temperature_C"].to_numpy()
    h = 5/60  # hours; temperatures are in degrees Celsius
    n = len(y)-1
    print(f"Station: {segment['Station'].iloc[0]}")
    print(f"Interval: {segment['timestamp'].iloc[0]} to {segment['timestamp'].iloc[-1]}")
    print(f"Samples={len(y)}, subintervals={n}, h={h:.6f} hour, duration={n*h:.1f} hours")
    print(f"Mean temperature: {y.mean():.4f} °C")
    print(f"Minimum: {y.min():.2f} °C at {segment.iloc[y.argmin()]['timestamp']}")
    print(f"Maximum: {y.max():.2f} °C at {segment.iloc[y.argmax()]['timestamp']}")

    values = {
        "Composite Trapezoidal": trapezoidal(y, h),
        "Composite Simpson 1/3": simpson_13(y, h),
        "Composite Simpson 3/8": simpson_38(y, h),
    }
    for name, value in values.items():
        print(f"{name}: {value:.6f} °C·h")

    segment[["timestamp", "temperature_C"]].to_csv("assignment_ix_cwprs_cleaned_24h.csv", index=False)
    plt.figure(figsize=(10, 4.8))
    plt.plot(segment["timestamp"], y, linewidth=1.2)
    plt.xlabel("Timestamp"); plt.ylabel("Air temperature (°C)")
    plt.title("CWPRS Campus, Pune: selected 24-hour temperature interval")
    plt.grid(True, alpha=0.3); plt.xticks(rotation=30, ha="right")
    plt.tight_layout(); plt.savefig("assignment_ix_temperature_timeseries.png", dpi=180); plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.bar(list(values.keys()), list(values.values()))
    plt.ylabel("Integral (°C·h)"); plt.title("Numerical integration method comparison")
    plt.xticks(rotation=12, ha="right"); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig("assignment_ix_method_comparison.png", dpi=180); plt.close()

if __name__ == "__main__":
    main()
