import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/predictive_maintenance.csv")

df = df[
    [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure"
    ]
]

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Machine failure")
plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure (0 = No Failure, 1 = Failure)")
plt.ylabel("Number of Machines")
plt.show()

plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Machine failure", y="Air temperature [K]")
plt.title("Air Temperature vs Machine Failure")
plt.xlabel("Machine Failure")
plt.ylabel("Air Temperature (K)")
plt.show()

plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Machine failure", y="Torque [Nm]")
plt.title("Torque vs Machine Failure")
plt.xlabel("Machine Failure")
plt.ylabel("Torque (Nm)")
plt.show()

plt.figure(figsize=(9, 6))
correlation = df.corr(numeric_only=True)
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()