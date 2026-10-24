"""Command line: python -m handui [run|screen|brain|record|train|labels|forget]"""
from __future__ import annotations

import argparse


def main() -> None:
    run_opts = argparse.ArgumentParser(add_help=False)  # shared so `python -m handui --bridge` works without `run`
    run_opts.add_argument("--camera", type=int, default=0)
    run_opts.add_argument("--width", type=int, default=1280)
    run_opts.add_argument("--height", type=int, default=720)
    run_opts.add_argument("--bridge", type=int, nargs="?", const=8765, default=None, metavar="PORT",
                          help="also stream events and hand state over ws://127.0.0.1:PORT (default 8765)")

    parser = argparse.ArgumentParser(prog="python -m handui", description="Hand-tracking gesture UI",
                                     parents=[run_opts])
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run", help="start the gesture UI (default)", parents=[run_opts])

    from .screening.simulate import PROFILES
    p_screen = sub.add_parser("screen", help="MotorLens: guided hand-function tests with a browser dashboard")
    p_screen.add_argument("--camera", type=int, default=0)
    p_screen.add_argument("--port", type=int, default=8765)
    p_screen.add_argument("--simulate", choices=sorted(PROFILES), metavar="PROFILE",
                          help=f"no camera; synthetic hand ({', '.join(sorted(PROFILES))}), kept apart from real results")
    p_screen.add_argument("--no-open", action="store_true", help="don't open the dashboard in a browser")
    p_screen.add_argument("--no-preview", action="store_true", help="don't show the camera window")

    p_brain = sub.add_parser("brain", help="Brain Explorer: EEG input on a 3D brain, navigated by hand")
    p_brain.add_argument("--eeg", metavar="FILE", help="preload a CSV or JSON of 10-20 channel values (one row per input)")
    p_brain.add_argument("--fs", type=float, metavar="HZ", help="rows are raw samples at this rate; use RMS per 0.5 s")
    p_brain.add_argument("--camera", type=int, default=0)
    p_brain.add_argument("--port", type=int, default=8765)
    p_brain.add_argument("--no-camera", action="store_true", help="mouse and keyboard only")
    p_brain.add_argument("--no-open", action="store_true", help="don't open the viewer in a browser")
    p_brain.add_argument("--no-preview", action="store_true", help="don't show the camera window")
    p_brain.add_argument("--adhd-api", default="http://127.0.0.1:8011", help="ADHD model server")
    p_brain.add_argument("--parkinsons-api", default="http://127.0.0.1:8012", help="Parkinson's model server")

    p_rec = sub.add_parser("record", help="record samples for a custom gesture")
    p_rec.add_argument("label", help="gesture name, e.g. thumbs_up (use 'none' for background poses)")
    p_rec.add_argument("--seconds", type=float, default=8.0)
    p_rec.add_argument("--camera", type=int, default=0)

    p_train = sub.add_parser("train", help="train the custom gesture classifier")
    p_train.add_argument("--augment", type=int, default=3, help="augmented copies per sample")

    sub.add_parser("labels", help="list recorded gesture labels")

    p_forget = sub.add_parser("forget", help="delete all samples for a label")
    p_forget.add_argument("label")

    args = parser.parse_args()

    if args.command in (None, "run"):
        from .app import run
        run(getattr(args, "camera", 0), getattr(args, "width", 1280), getattr(args, "height", 720),
            getattr(args, "bridge", None))
    elif args.command == "screen":
        from .screening.runner import run_screening
        run_screening(args.camera, args.simulate, args.port, not args.no_open, not args.no_preview)
    elif args.command == "brain":
        from .brain.eeg import load
        from .brain.runner import run_brain
        try:
            recording = load(args.eeg, fs=args.fs) if args.eeg else None
        except (OSError, ValueError) as e:
            parser.exit(1, f"Could not read EEG: {e}\n")
        run_brain(recording, args.camera, args.port, not args.no_open, not args.no_preview, not args.no_camera,
                  {"adhd": args.adhd_api, "parkinsons": args.parkinsons_api})
    elif args.command == "record":
        from .record import record
        record(args.label, args.seconds, args.camera)
    elif args.command == "train":
        from .classifier import train
        try:
            report = train(augment=args.augment)
        except ValueError as e:
            parser.exit(1, f"{e}\n")
        for label, n in sorted(report["counts"].items()):
            print(f"  {label:<20} {n:>5} samples")
        if report["holdout_accuracy"] is not None:
            print(f"Held-out accuracy: {report['holdout_accuracy']:.1%}")
        print(f"Saved model to {report['model_path']}")
    elif args.command == "labels":
        from .classifier import label_counts
        counts = label_counts()
        if not counts:
            print("No samples yet. Try: python -m handui record thumbs_up")
        for label, n in sorted(counts.items()):
            print(f"  {label:<20} {n:>5} samples")
    elif args.command == "forget":
        from .classifier import remove_label
        print(f"Removed {remove_label(args.label)} samples for '{args.label}'.")


if __name__ == "__main__":
    main()
