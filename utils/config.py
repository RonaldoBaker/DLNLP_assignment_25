import os
import argparse

"""
CLI arguments for the entire repo
"""

parser = argparse.ArgumentParser(description="CLI arguments for the entire repo")

# for base_main.py and multisource_main.py
parser.add_argument("--GPU", type=int, default="0", help="GPU id to use")
parser.add_argument("-SFP", "--SAVE_FILEPATH", type=str, default="/scratch/zceerba/nlp/DLNLP_assignment_25/", help="Path to save figures")
parser.add_argument("--LOG_PATH", type=str, default="/scratch/zceerba/nlp/DLNLP_assignment_25/logs/", help="Path to save logs")
parser.add_argument("--MODEL", type=str, default="single", choices=["single", "multi"], help="Model type to use")
parser.add_argument("-D", "--DROPOUT", type=float, default=0.2, help="Dropout rate")

# for multisource_main.py
parser.add_argument("-FT", "--FUSION_TYPE", type=str, default="single", choices=["single", "multi"], help="Fusion type for multisource transformer")
parser.add_argument("-TKNS", "--TOKENISATIONS", nargs="+", type=str, default=["word"], choices=["word", "subword", "syllable", "char"], help="Tokenisation to use for the multisource transformer")

config = parser.parse_args()
