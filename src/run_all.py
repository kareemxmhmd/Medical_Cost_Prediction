import logging
import os
import subprocess
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eda import generate_eda_figures
from src.models.train_model import train_and_evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("pipeline_runner")


def main() -> None:
    """Executes the complete end-to-end data, modeling, evaluation, and test pipeline."""
    logger.info("=== Starting Medical Cost Prediction Pipeline ===")
    
    # 1. Generate EDA Diagnostic Figures
    logger.info("[1/3] Generating EDA diagnostic figures...")
    generate_eda_figures()
    
    # 2. Run Model Benchmark and Champion Quantile Training
    logger.info("[2/3] Executing model benchmarking & champion training...")
    metrics = train_and_evaluate()
    
    # 3. Execute Pytest Test Suite
    logger.info("[3/3] Executing automated test suite (pytest)...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent.resolve())
    
    test_result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"],
        env=env
    )
    if test_result.returncode != 0:
        logger.error("Automated test suite failed.")
        sys.exit(1)
        
    logger.info("=== Pipeline Completed Successfully ===")
    logger.info("Champion Metrics: MAE=$%s, RMSE=$%s, R2=%s, 80%% Interval Coverage=%s",
                metrics["metrics"]["mae"],
                metrics["metrics"]["rmse"],
                metrics["metrics"]["r2"],
                metrics["metrics"]["coverage_80"])
    logger.info("Artifacts available in 'artifacts/' and 'reports/'")


if __name__ == "__main__":
    main()

