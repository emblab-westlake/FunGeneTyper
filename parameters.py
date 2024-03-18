# -*-coding:utf-8-*-
import argparse
# load
def params_parser():
    parser = argparse.ArgumentParser(description="ESM-1b pretraining hyper-parameters")
    parser.add_argument(
        "--layers", default=33, type=int, metavar="N", help="number of layers"
    )
    parser.add_argument(
        "--embed_dim", default=1280, type=int, metavar="N", help="embedding dimension"
    )
    parser.add_argument(
        "--logit_bias", action="store_true", help="whether to apply bias to logits"
    )
    parser.add_argument(
        "--ffn_embed_dim",
        default=5120,
        type=int,
        metavar="N",
        help="embedding dimension for FFN",
    )
    parser.add_argument(
        "--dropout",
        default=0.0,
        type=float,
        help="Dropout to apply."
    )
    parser.add_argument(
        "--attention_heads",
        default=20,
        type=int,
        metavar="N",
        help="number of attention heads",
    )
    parser.add_argument(
        "--arch",
        default="roberta_large",
        type=str,
        help="model architecture",
    )
    parser.add_argument(
        "--max_positions",
        default=1024,
        type=int,
        help="max positions",
    )

    # classification
    parser.add_argument(
        "--input",
        type=str,
        default="example/Resistance_gene/test.fasta"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=3
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="Using GPU's id, default using GPU 0"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="example/Resistance_gene/output_class.txt"
    )
    parser.add_argument(
        "--nogpu",
        action="store_true",
        help="Do not use GPU even if available"
    )
    parser.add_argument(
        "--topK",
        type=int,
        default=1
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="ARGs",
        help="adapter name"
    )
    parser.add_argument(
        "--group",
        action="store_true",
        help="Whether group category classification is required"
    )

    # Train
    parser.add_argument(
        "--Train_Category",
        type=int,
        default=20
    )

    # multi-GPU
    parser.add_argument(
        "--local_rank",
        default=-1,
        type=int,
        help="node rank for distributed testing",
    )

    return parser.parse_args()