import argparse
import json

from sentinelml.modeling import train, recalibrate_monitor
from sentinelml.experiments import study, compare_candidate, promote, benchmark_alerts, benchmark_shift_sweep
from sentinelml.service import Monitor
from sentinelml.simulation import simulate


def main():
    parser = argparse.ArgumentParser(description="SentinelML local research platform")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train")
    sub.add_parser("recalibrate-monitor")
    benchmark = sub.add_parser("benchmark-alerts")
    benchmark.add_argument("--batches", type=int, default=500)
    benchmark.add_argument("--size", type=int, default=500)
    sweep = sub.add_parser("benchmark-shift-sweep")
    sweep.add_argument("--episodes", type=int, default=100)
    sweep.add_argument("--stable-batches", type=int, default=500)
    sweep.add_argument("--size", type=int, default=500)
    sweep.add_argument("--null-draws", type=int)
    experiment = sub.add_parser("experiment")
    experiment.add_argument("--repetitions", type=int, default=30)
    sub.add_parser("demo")
    sub.add_parser("audit")
    compare = sub.add_parser("compare")
    compare.add_argument("--scenario", default="repayment_shift")
    compare.add_argument("--strength", type=float, default=1.0)
    promotion = sub.add_parser("promote")
    promotion.add_argument("version")
    promotion.add_argument("--approve", action="store_true")
    args = parser.parse_args()
    if args.command == "train":
        result = train()
    elif args.command == "recalibrate-monitor":
        result = recalibrate_monitor()
    elif args.command == "benchmark-alerts":
        result = benchmark_alerts(args.batches, args.size)
    elif args.command == "benchmark-shift-sweep":
        full = benchmark_shift_sweep(args.episodes, args.size, args.stable_batches, args.null_draws)
        result = {k: v for k, v in full.items() if k != "groups"}
    elif args.command == "experiment":
        full = study(args.repetitions)
        result = {k: v for k, v in full.items() if k != "groups"}
    elif args.command == "compare":
        full = compare_candidate(args.scenario, args.strength)
        result = {k: v for k, v in full.items() if not k.endswith("_ids")}
    elif args.command == "promote":
        result = promote(args.version, args.approve)
    elif args.command == "audit":
        from sentinelml.audit import audit
        result = audit()
    else:
        monitor = Monitor()
        result = []
        for index, scenario in enumerate(["stable", "age_shift", "repayment_shift", "missingness", "label_shift"]):
            x, y, source_ids = simulate(monitor.bundle["parts"]["simulation"], scenario, seed=900 + index)
            report = monitor.ingest(x, simulation_info={"scenario": scenario, "strength": 1.0,
                                                      "seed": 900 + index, "source_ids": source_ids})
            # Leave the last batch unlabelled to illustrate delayed labels.
            if scenario != "label_shift":
                report = monitor.label(report["batch_id"], {str(i): int(v) for i, v in enumerate(y)})
            result.append({"scenario": scenario, "batch_id": report["batch_id"], "label_status": report["label_status"]})
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
