"""Fine-tune Laya on a small GPU by training only the top of the encoder.

    python finetune.py pacman/data/train.jsonl ./checkpoints/pacman --top-layers 6

A full fine-tune of the 421M English checkpoint needs about 7 GB just for the
weights, gradients and AdamW state, before a single activation. That doesn't fit
on a 6 GB card. Freezing the whole encoder does fit, but Laya's own docs say it
"gains much less than a full fine-tune".

The middle ground: freeze the embeddings and the bottom of the encoder, train
the top N layers and the decision head. `laya.train` already skips parameters
that don't require gradients, so this is a wrapper, not a fork: we freeze layers
right after the checkpoint loads and let `finetune()` do everything else
(training, calibration on a held-out slice, saving a checkpoint `laya.load` opens).
"""
import argparse
import json
import time

import laya.train as lt


def freeze_bottom(top_layers: int):
    original = lt.load_checkpoint

    def load_and_freeze(model_dir):
        model, tok, cfg = original(model_dir)
        layers = model.encoder.layers
        keep = len(layers) - top_layers
        for p in model.encoder.embeddings.parameters():
            p.requires_grad_(False)
        for layer in layers[:keep]:
            for p in layer.parameters():
                p.requires_grad_(False)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"training the top {top_layers} of {len(layers)} encoder layers + head: "
              f"{trainable / 1e6:.0f}M of {total / 1e6:.0f}M parameters", flush=True)
        return model, tok, cfg

    lt.load_checkpoint = load_and_freeze


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="a JSONL of {state, questions, gold|expected} rows, or a CSV")
    ap.add_argument("out")
    ap.add_argument("--base", default="english", help="english, multilingual, typed-decisions, a Hub id or a directory")
    ap.add_argument("--top-layers", type=int, default=6, help="encoder layers to train; -1 trains all of them")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--loss", default="rlcd", choices=["rlcd", "soft-ce"])
    ap.add_argument("--max-len", type=int)
    ap.add_argument("--eval-data")
    ap.add_argument("--shuffle-options", action="store_true")
    ap.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    csv = ap.add_argument_group("CSV input (one labelled text per row)")
    csv.add_argument("--text-column", default="text")
    csv.add_argument("--label-column", default="label")
    csv.add_argument("--question-id", default="label", help="the id the question is saved under in questions.json")
    csv.add_argument("--instructions", help="the question to ask, e.g. 'Which team should handle this?'")
    args = ap.parse_args()

    if args.top_layers >= 0:
        freeze_bottom(args.top_layers)
    config = lt.TrainConfig(
        epochs=args.epochs,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        loss=args.loss,
        max_len=args.max_len,
        eval_data=args.eval_data,
        shuffle_options=("choice",) if args.shuffle_options else (),
        text_column=args.text_column,
        label_column=args.label_column,
        question_id=args.question_id,
        instructions=args.instructions,
    )
    base = lt.resolve_checkpoint_dir(args.base)
    t = time.time()
    report = lt.finetune(args.data, base, args.out, config, device=args.device)
    report["minutes"] = round((time.time() - t) / 60, 1)
    keys = ["train_items", "calibration_items", "skipped", "epoch_loss", "temperature", "minutes"]
    print(json.dumps({k: report.get(k) for k in keys}, indent=2, default=str))


if __name__ == "__main__":
    main()
