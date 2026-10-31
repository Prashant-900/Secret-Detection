"""One entry point; heavy model dependencies load only for model commands."""
import importlib
import sys

COMMANDS = {
    "ingest": "ingest", "audit": "audit", "generate": "generate",
    "candidates": "candidates", "evaluate": "evaluate", "train": "train",
    "predict": "predict", "dataset": "workflow", "experiments": "experiments",
    "scan": "scan", "smoke": "smoke", "smoke-train": "smoke_train",
    "doctor": "doctor", "mine": "mine", "assess": "assess", "baselines": "baselines",
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print("usage: python -m secrets_pipeline COMMAND [options]\n\nCommands: " + ", ".join(COMMANDS))
        return 0
    command = args.pop(0)
    if command not in COMMANDS:
        raise SystemExit("Unknown command: " + command)
    return importlib.import_module("." + COMMANDS[command], __package__).main(args)


if __name__ == "__main__":
    raise SystemExit(main())
