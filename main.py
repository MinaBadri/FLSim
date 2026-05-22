import sys
from utils import (
    load_config,
    build_data_pipeline,
    build_hardware_profiles,
    build_fleet,
    build_registry,
    build_server,
)


def main(config_path: str):
    print("=" * 55)
    print(" FL Simulation — Vehicular Churn")
    print("=" * 55 + "\n")

    # Load config
    print(f"Loading config: {config_path}")
    cfg = load_config(config_path)

    # Confirm key settings
    print(f"  num_clients     : {cfg['simulation']['num_clients']}")
    print(f"  num_rounds      : {cfg['simulation']['num_rounds']}")
    print(f"  drop_prob       : {cfg['churn']['drop_prob']}")
    print(f"  strategy        : {cfg['aggregation']['strategy']}")
    print(f"  learning_rate   : {cfg['training']['learning_rate']}\n")

    # Build all components
    print("Building data pipeline ...")
    client_loaders, test_loader, client_indices = build_data_pipeline(cfg)

    print("Building hardware profiles ...")
    hardware_profiles = build_hardware_profiles(cfg)

    print("Building car fleet ...")
    fleet = build_fleet(cfg, client_loaders, hardware_profiles)

    print("Building client registry ...")
    registry = build_registry(cfg, fleet)

    print("Building FL server ...\n")
    server = build_server(cfg, registry, test_loader)

    # Run
    history = server.run()

    print(f"\nDone. {len(history)} rounds logged.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <config_path>")
        print("Example: python main.py configs/sanity.yaml")
        sys.exit(1)

    main(sys.argv[1])