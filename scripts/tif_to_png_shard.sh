# This script shards a large directory of TIFF files into 
# smaller directories of PNG files using vips. It uses
# 256 buckets from 00-ff and buckets files using the
# first two chars of the filename's MD5 hash. 
# Useful for large datasets, makes them compatible with 
# objects like PyTorch's DataLoaders or ImageFolder

#!/usr/bin/env bash
#SBATCH --job-name=tif_to_png
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Required positional args
IN_DIR="${1:?Usage: sbatch $0 <in_dir> <out_dir> <conda_env>}"
OUT_DIR="${2:?Usage: sbatch $0 <in_dir> <out_dir> <conda_env>}"
CONDA_ENV="${3:?Usage: sbatch $0 <in_dir> <out_dir> <conda_env>}"

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Workers: ${SLURM_CPUS_PER_TASK:-8}"
echo "Input:   $IN_DIR"
echo "Output:  $OUT_DIR"

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [ -z "$CONDA_BASE" ]; then
  echo "ERROR: conda not found in PATH"; exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

mkdir -p "$OUT_DIR"
P="${SLURM_CPUS_PER_TASK:-8}"

find "$IN_DIR" -type f -name '*.tif' -print0 | \
  xargs -0 -P "$P" -n 1 bash -c '
    set -euo pipefail
    f="$0"
    b="$(basename "${f%.tif}")"
    shard="$(printf "%s" "$b" | md5sum | cut -c1-2)"
    outdir="'"$OUT_DIR"'/$shard"
    mkdir -p "$outdir"
    vips copy "$f" "$outdir/$b.png"
  '

echo "Done: $(date)"