import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpld3

# Generate synthetic data
np.random.seed(42)
latencies = np.concatenate([
    np.random.normal(10, 2, 800),   # Fast responses (P50)
    np.random.normal(20, 5, 150),   # Moderate latency (P90)
    np.random.normal(50, 10, 50)    # High latency (P99)
])
latencies = np.clip(latencies, 5, 100)  # Ensure valid range
inference_times = latencies + np.random.normal(2, 1, len(latencies))

# Create DataFrame
df = pd.DataFrame({
    "Latency (ms)": latencies,
    "Inference Time (ms)": inference_times,
})

# Plot
plt.figure(figsize=(5, 3))
sns.boxplot(x=pd.qcut(df["Latency (ms)"], q=[0, 0.5, 0.9, 0.99, 1.0]), y=df["Inference Time (ms)"])
plt.xlabel("Latency Percentiles")
plt.ylabel("Model Inference Time (ms)")
plt.title("Impact of Latency on Model Inference Time")

html_filename = __file__.replace('.py', '.html')
mpld3.save_html(plt.gcf(), html_filename)
