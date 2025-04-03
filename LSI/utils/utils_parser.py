import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Process optional float inputs.")
    parser.add_argument("--noise", type=float, default=1.0, help="Noise scale for differential privacy")
    parser.add_argument("--clip", type=float, default=1.0, help="Clipping bound for differential privacy")
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--n_rem", type=int, default=0)
    parser.add_argument("--kl_every_n_epochs", type=int, default=0)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--name_ext", type=str, default="corrupted_idx_human_annotated")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--mom", type=float, default=0.9)
    parser.add_argument("--wd", type=float, default=1e-4, help="Batch size for training")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--model", type=str, default="TinyModel")
    parser.add_argument("--subset", type=int, default=50)
    parser.add_argument("--range1", type=int, default=0)
    parser.add_argument("--range2", type=int, default=0)
    parser.add_argument("--laptype", type=str, default="diag")
    parser.add_argument("--corrupt_label", type=float, default=0.0)
    parser.add_argument("--corrupt_data", type=float, default=0.0)
    parser.add_argument("--corrupt_data_label", type=int, nargs='+', default=[1, 2], help="List of labels to corrupt")
    parser.add_argument("--human_label_noise", type=bool, default=False)
    parser.add_argument("--make_private", type=bool, default=False)
    parser.add_argument("--use_sam", action="store_true", help="Use SAM optimizer for training")
    parser.add_argument("--is_compressed", action="store_true", help="Use compressed dataset")
    parser.add_argument("--compress_model", type=str, default="ResNet18", help="Model to use for compression")
    parser.add_argument("--data_path", type=str, default="./LSI/Datasets/data_dir")
    parser.add_argument("--active_classes", type=int, nargs='+', default=[], help="List of active classes to use")
    parser.add_argument("--json_path", type=str, default="./LSI/Datasets/datasets.json")
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--sub_dir", type=str, default="regular_LSI")
    parser.add_argument("--noisy_label_location", type=str, default="./LSI/Datasets/mislabeled_cifarh/")

    return parser

def generate_experiment_description(args):
    description = ["\n\n"]
    description.append(f"   ##       ######   ##   ")
    description.append(f"   ##       ##       ##   ")
    description.append(f"   ##       ######   ##   ")
    description.append(f"   ##           ##   ##   ")
    description.append(f"   ######   ######   ##   ")
    # General settings
    description.append(f"General Settings:")
    if args.name:
        description.append(f"Experiment Name: {args.name}")
    if args.name_ext:
        description.append(f"Name Extension: {args.name_ext}")
    description.append(f"LSI Probe Model: {args.model}")
    description.append(f"Number of Seeds: {args.n_seeds}")
    description.append(f"Epochs: {args.epochs}")
    description.append(f"Learning Rate: {args.lr}")
    description.append(f"Momentum: {args.mom}")
    description.append(f"Weight Decay: {args.wd}")
    description.append(f"\n")

    # Dataset settings
    description.append(f"Dataset Settings:")
    description.append(f"Dataset: {args.dataset}")
    if args.subset:
        description.append(f"Subset Size: {args.subset}")
    if args.active_classes:
        description.append(f"Active Classes: {args.active_classes}")
    if args.is_compressed:
        description.append(f"Compressed Dataset: {args.is_compressed}")
    description.append(f"Data Path: {args.data_path}")
    description.append(f"JSON Path: {args.json_path}")
    description.append(f"\n")

    # Corruption settings
    description.append(f"Corruption Settings:")
    corrupt = False
    if args.corrupt_label > 0:
        description.append(f"Corrupt Label Probability: {args.corrupt_label}")
        corrupt = True
    if args.corrupt_data > 0:
        description.append(f"Corrupt Data Probability: {args.corrupt_data}")
        description.append(f"Corrupt Data Labels: {args.corrupt_data_label}")
        corrupt = True
    if args.human_label_noise:
        description.append(f"Human Label Noise: {args.human_label_noise}")
        description.append(f"Noisy Label Location: {args.noisy_label_location}")
        corrupt = True
    if args.use_sam:
        description.append(f"Using SAM Optimizer: {args.use_sam}")
        corrupt = True
    if not corrupt:
        description.append(f"No Corruption Applied")
    description.append(f"\n")

    # Model settings
    description.append(f"Compression Model Settings:")
    description.append(f"Compression Model: {args.compress_model}")
    description.append(f"\n")

    # Privacy settings
    description.append(f"Privacy Settings:")
    if args.make_private:
        description.append(f"Private Training Enabled")
        description.append(f"Noise: {args.noise}")
        description.append(f"Clipping: {args.clip}")
    else:
        description.append(f"Private Training Disabled")
    description.append(f"\n")

    # Save settings
    description.append(f"Save Settings:")
    description.append(f"Save Directory: {args.save_dir}")
    description.append(f"Sub Directory: {args.sub_dir}")

    # Print the experiment description
    print("\n".join(description))