import os

from config import get_output_dir
from data import load_data, flag_data
from evaluation import _calculate_metrics, save_metrics
from utils import generate_model_name, save_json


def main_aoflagger(config_vals: dict):
    """
    Runs analysis with AOFlagger instead of NN appraoch
    """

    config_vals["model_name"] = generate_model_name(config_vals)
    print(config_vals["model_name"])
    output_dir = os.path.join(
        get_output_dir(),
        config_vals["model_type"],
        config_vals["anomaly_type"],
        config_vals["model_name"],
    )
    if config_vals["dataset"] == "TABASCAL":
        _, _, test_x, test_y, _ = load_data(
            config_vals,
            num_sat=config_vals["satellite"],
            num_ground=config_vals["ground_source"],
        )
    else:
        _, _, test_x, test_y, _ = load_data(config_vals)
    # Evaluate with AOFlagger
    flags = flag_data(
        test_x, threshold=config_vals["threshold"], mode=config_vals["dataset"]
    ).astype("float32")
    # Calculate metrics
    metrics = _calculate_metrics(test_y, flags)
    print(metrics)
    # Save json
    os.makedirs(output_dir, exist_ok=True)
    save_json(config_vals, output_dir, "config")
    save_metrics(
        {},
        metrics,
        {},
        {},
        config_vals["model_type"],
        config_vals["anomaly_type"],
        config_vals["model_name"],
    )


if __name__ == "__main__":
    config_vals = {
        "num_layers": 1,
        "latent_dimension": 0,
        "threshold": 10,
        "anomaly_type": "MISO",
        "dataset": "HERA",
        "model_type": "AOFLAGGER",
        "excluded_rfi": None,
    }
    threshold_range = [0.5, 1, 3, 5, 7, 9, 10, 20, 50, 100, 200]
    excluded_rfi = [None, "rfi_stations", "rfi_dtv", "rfi_impulse", "rfi_scatter"]
    satellite_range = [0, 1, 2]
    ground_source_range = [0, 1, 3]
    # HERA
    for threshold in threshold_range:
        config_vals["threshold"] = threshold
        main_aoflagger(config_vals)
    for rfi in excluded_rfi:
        config_vals["excluded_rfi"] = rfi
        main_aoflagger(config_vals)
    # LOFAR
    config_vals["excluded_rfi"] = None
    config_vals["dataset"] = "LOFAR"
    config_vals["threshold"] = 10
    main_aoflagger(config_vals)
    # TABASCAL
    config_vals["dataset"] = "TABASCAL"
    for satellite in satellite_range:
        config_vals["satellite"] = satellite
        for ground_source in ground_source_range:
            config_vals["ground_source"] = ground_source
            main_aoflagger(config_vals)
