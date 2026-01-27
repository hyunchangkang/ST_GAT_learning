# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from .runner import infer_from_yaml

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="YAML path (default: params.yaml or params_tw.yaml)")
    args = parser.parse_args()
    infer_from_yaml(args.config)

if __name__ == "__main__":
    main()
