import hashlib
import os
from argparse import ArgumentParser

import fsspec

parser = ArgumentParser()
parser.add_argument('--output-dir', default="deduped")
parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')
args = parser.parse_args()

common_prefix = os.path.commonprefix(args.files) if len(args.files) > 1 else os.path.dirname(args.files[0])

os.makedirs(args.output_dir, exist_ok=True)


def hash(item): return hashlib.sha256(str(item).encode('utf-8')).hexdigest()


seen = set()
for f in args.files:
    removed = 0
    with fsspec.open(f, "r", compression="infer") as input:
        out_name = os.path.join(args.output_dir, os.path.relpath(f, common_prefix))
        print(f"Deduplicating {f} to {out_name}...")
        with fsspec.open(out_name, "w", compression="infer") as output:
            for line in input:
                h = hash(line)
                if h not in seen:
                    output.write(line)
                    seen.add(h)
                else:
                    removed += 1

    print(f"Removed {removed} duplicates from {f}")

print("Deduplication complete.")
