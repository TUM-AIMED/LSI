import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Process optional float inputs.")
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--n_rem", type=int, default=0)
    parser.add_argument("--batch_num", type=int, default=1)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--name_ext", type=str, default="corrupted_idx_human_annotated")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--mom", type=float, default=0.9)
    parser.add_argument("--dataset", type=str, default="cifar10compressed")
    parser.add_argument("--model", type=str, default="TinyModel")
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--range1", type=int, default=0)
    parser.add_argument("--range2", type=int, default=0)
    parser.add_argument("--laptype", type=str, default="diag")
    parser.add_argument("--corrupt", type=float, default=0.0)
    parser.add_argument("--corrupt_data", type=float, default=0.0)
    parser.add_argument("--corrupt_data_label", type=int, default=0)
    parser.add_argument("--human_label_noise", type=bool, default=True)
    parser.add_argument("--use_sam", action="store_true", help="Use SAM optimizer for training")
    return parser