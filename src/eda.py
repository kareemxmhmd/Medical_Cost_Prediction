import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("eda_generator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def generate_eda_figures(data_path: str = "data/insurance.csv", output_dir: str = "reports/figures") -> None:
    """Generates distribution, interaction, and correlation diagnostic figures for medical claims."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Reading dataset for EDA analysis from %s", data_path)
    df = pd.read_csv(data_path)
    
    # 1. Target Distribution (Raw vs. Log-Transformed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df["charges"], kde=True, ax=axes[0], color="#1f77b4")
    axes[0].set_title("Raw Charges Distribution (Right-Skewed)")
    axes[0].set_xlabel("Charges ($)")
    
    sns.histplot(np.log1p(df["charges"]), kde=True, ax=axes[1], color="#2ca02c")
    axes[1].set_title("Log-Transformed Charges Distribution (Normalized)")
    axes[1].set_xlabel("log(1 + Charges)")
    plt.tight_layout()
    fig.savefig(out_path / "charges_distribution.png", dpi=150)
    plt.close(fig)
    
    # 2. Risk Driver Boxplots (Smoker, Region, Sex)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.boxplot(x="smoker", y="charges", hue="smoker", data=df, ax=axes[0], palette="Set2", legend=False)
    axes[0].set_title("Charges by Smoker Status")
    axes[0].set_ylabel("Charges ($)")
    
    sns.boxplot(x="region", y="charges", hue="region", data=df, ax=axes[1], palette="Set2", legend=False)
    axes[1].set_title("Charges by Region")
    axes[1].set_ylabel("")
    
    sns.boxplot(x="sex", y="charges", hue="sex", data=df, ax=axes[2], palette="Set2", legend=False)
    axes[2].set_title("Charges by Sex (Negligible Variance)")
    axes[2].set_ylabel("")
    plt.tight_layout()
    fig.savefig(out_path / "charges_boxplots.png", dpi=150)
    plt.close(fig)
    
    # 3. BMI vs Charges Non-Linear Smoker Interaction
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="bmi", y="charges", hue="smoker", data=df, alpha=0.7, palette=["#2ca02c", "#d62728"])
    plt.title("BMI vs. Charges (Interaction with Smoker Status)")
    plt.xlabel("BMI")
    plt.ylabel("Charges ($)")
    plt.tight_layout()
    plt.savefig(out_path / "bmi_vs_charges_smoker.png", dpi=150)
    plt.close()
    
    # 4. Numerical Covariate Correlation Matrix
    plt.figure(figsize=(8, 6))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Numerical Covariate Correlation Matrix")
    plt.tight_layout()
    plt.savefig(out_path / "correlation_matrix.png", dpi=150)
    plt.close()
    
    logger.info("EDA figures successfully generated in '%s'", output_dir)


if __name__ == "__main__":
    generate_eda_figures()

