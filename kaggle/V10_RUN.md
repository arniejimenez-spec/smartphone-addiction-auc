# Run v10 on Kaggle

V10 requires a Kaggle GPU because the exact regime matrix has 1,653 float64
columns and the optimizer holds the full training matrix on the accelerator.

## 1. Create the private pool-cache dataset

Create a new private Kaggle dataset and upload these three local files from
`artifacts/v8/full/cache`:

- `public_pool_oof.npy`
- `public_pool_test.npy`
- `public_pool_manifest.json`

Do not upload `train.csv` or `test.csv`; the competition supplies those.

## 2. Import and configure the notebook

Import `kaggle/v10_exact_gpu_fusion.ipynb` as a Kaggle notebook. Then:

1. Add the `playground-series-s6e8` competition data.
2. Add the private pool-cache dataset from step 1.
3. Enable a GPU accelerator. A 16 GB T4 is the minimum target.
4. Run all cells.

The notebook auto-discovers both inputs. It writes checkpoints after every
optimizer block, plus convergence diagnostics in `result_v10.json`.

## 3. Download the selected output

Download `submission_v10.csv` from the notebook output. This is the exact
reference-style 70% fusion / 30% base blend. The notebook also writes
`submission_v10_mix.csv`, the unblended rank-logit/regime alternative.

The selected submission is not created unless both the dual and regime models
meet a numerical convergence criterion. If the run reaches 4,000 iterations,
save the checkpoint outputs and rerun with a larger `--max-total-iter` value.
