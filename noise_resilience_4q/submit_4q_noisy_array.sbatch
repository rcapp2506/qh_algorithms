#!/usr/bin/env bash
# submit_4q_noisy_array.sbatch — Wave-K 4q noisy multiseed (array job 10×)
#
# Lancia 10 task SLURM paralleli (uno per ogni seed in [42..51]) che girano
# il training 4q noisy con NoiseModel ibm_fez restricted a qubit 0-3.
#
# Each task writes results_seed_<seed>.json into $OUTPUT_DIR. After all
# tasks have completed, download the 10 JSON files to the local machine and run:
#   python project_4q_to_9q.py  --noisy-dir results_4q_noisy --noiseless-dir results_4q_noiseless
#   python wilcoxon_cross_arch.py --runs ...
#
# USAGE:
#   cd /gpfs/ddn/cta/cappu/Q-CONV/slurm
#   sbatch submit_4q_noisy_array.sbatch
#
# WALL TIME: 2h per task (atteso reale 30-90 min/seed)

#SBATCH --job-name=qcnn_4q_noisy
#SBATCH --account=default
#SBATCH --partition=levante
#SBATCH --array=0-9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/ddn/cta/cappu/Q-CONV/slurm/logs/4q_noisy_%A_%a.out
#SBATCH --error=/gpfs/ddn/cta/cappu/Q-CONV/slurm/logs/4q_noisy_%A_%a.err

WORK_DIR="/gpfs/ddn/cta/cappu/Q-CONV"
CODE_DIR="$WORK_DIR/code"
DATASET_DIR="$WORK_DIR/dataset"
NOISE_SNAPSHOT="$WORK_DIR/noise_models/ibm_fez_20260516.pkl"
OUTPUT_DIR="$WORK_DIR/output/4q_noisy_multiseed"
mkdir -p "$OUTPUT_DIR"

# Seed: base_seed + array index → [42, 43, ..., 51]
BASE_SEED=42
SEED=$((BASE_SEED + SLURM_ARRAY_TASK_ID))

echo "═══════════════════════════════════════════════════════════════"
echo "  4q QCNN noisy — array task $SLURM_ARRAY_TASK_ID, seed $SEED"
echo "═══════════════════════════════════════════════════════════════"
echo "Hostname:    $(hostname)"
echo "Date:        $(date -Iseconds)"
echo "Job:         ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID})"
echo "CPUs alloc:  ${SLURM_CPUS_PER_TASK}"
echo "Seed:        $SEED"
echo ""

# ── Pre-flight ──
[ -f "$NOISE_SNAPSHOT" ] && echo "  ✓ noise snapshot" || { echo "  ✗ snapshot MANCANTE: $NOISE_SNAPSHOT"; exit 1; }
[ -d "$DATASET_DIR/training" ] && echo "  ✓ training dir" || { echo "  ✗ training dir MANCANTE"; exit 1; }
[ -d "$DATASET_DIR/validation" ] && echo "  ✓ validation dir" || { echo "  ✗ validation dir MANCANTE"; exit 1; }
[ -f "$CODE_DIR/run_4q.py" ] && echo "  ✓ run_4q.py" || { echo "  ✗ run_4q.py MANCANTE in $CODE_DIR"; exit 1; }
[ -f "$CODE_DIR/parallel_quanv_sim.py" ] && echo "  ✓ parallel_quanv_sim.py" || { echo "  ✗ parallel_quanv_sim.py MANCANTE"; exit 1; }
echo ""

# ── Env Python ──
module purge
module load python/3.13.5-gcc-11.5.0
source /gpfs/ddn/cta/cappu/qiskitq/bin/activate
echo "Python:      $(which python)  ($(python --version))"
echo ""

# ── Threading: BLAS=1 OBBLIGATORIO ──
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
echo "Threading: OMP=$OMP_NUM_THREADS, BLAS=1 (8 worker Aer, no oversubscription)"
echo "Python unbuffered: PYTHONUNBUFFERED=$PYTHONUNBUFFERED"
echo ""

# ── Run ──
cd "$CODE_DIR"
t0=$(date +%s)
python -u run_4q.py \
    --seed "$SEED" \
    --backend sim_noisy \
    --noise-snapshot "$NOISE_SNAPSHOT" \
    --noise-qubits 0 1 2 3 \
    --train-dir "$DATASET_DIR/training" \
    --val-dir "$DATASET_DIR/validation" \
    --output-dir "$OUTPUT_DIR" \
    --aer-parallel 8 \
    --max-samples 100 \
    --max-epochs 10 \
    --batch-size 16 \
    --shots 500

RC=$?
t_end=$(date +%s)
DT=$((t_end - t0))

echo ""
if [ $RC -eq 0 ]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ✓ SEED $SEED COMPLETED in ${DT}s ($((DT/60))m $((DT%60))s)"
    echo "═══════════════════════════════════════════════════════════════"
    ls -la "$OUTPUT_DIR/results_seed_$(printf '%03d' $SEED).json" 2>/dev/null
else
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ✗ SEED $SEED FAILED (exit $RC) after ${DT}s"
    echo "═══════════════════════════════════════════════════════════════"
fi
exit $RC
