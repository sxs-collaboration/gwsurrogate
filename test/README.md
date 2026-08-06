The tests folder includes unit tests, integration tests, and performance benchmarks for the `gwsurrogate` package. Please consult the individual test scripts for details on how to run them.

The regressions tests are run automatically on GitHub Actions for each pull request and branch. The performance benchmarks are run manually from GitHub Actions and can also be run directly on HPC machines. Please see the Performance Benchmarks section below for details on how to run the benchmarks.

# Performance Benchmarks

This directory includes the performance benchmark used to compare the runtime of surrogate model evaluations across two git references, such as the current branch and another branch or pull request.

The main benchmark script is `benchmark_surrogate_evaluations.py`. Please consult the script's docstring and command-line help for details on how to use it.

## Running In GitHub Actions

The benchmark is run manually from GitHub Actions:

1. Open the repository on GitHub.
2. Go to `Actions`.
3. Select `GWSurrogate Performance Benchmark`.
4. Click `Run workflow`.

The workflow accepts these inputs that are passed to the benchmark script. Please consult the script's docstring and command-line help for details on these options.

## Reports

Each successful run uploads an Actions artifact named `benchmark-results`. The artifact contains the HTML, JSON, Markdown, and plot outputs for that workflow run.

The benchmark landing page is published to GitHub Pages:

https://sxs-collaboration.github.io/gwsurrogate/benchmarks/

The latest successful benchmark report is published here:

https://sxs-collaboration.github.io/gwsurrogate/benchmarks/latest/

Archived benchmark reports, when any have been saved, are listed here:

https://sxs-collaboration.github.io/gwsurrogate/benchmarks/runs/

## Archiving Guidelines

Most benchmark runs should not be archived. By default, a run updates the latest GitHub Pages report and leaves the full output attached to the Actions run as an artifact.

Set `archive_report` to `true` when the run is worth keeping as a long-term reference. Good candidates include release baselines, important pull request comparisons, major dependency changes, major surrogate evaluation changes, and confirmed performance regressions or fixes.

## Running On HPC Machines

The benchmark can also be run directly on an HPC system. The exact module and conda setup will vary by system, but the workflow on a machine such as Anvil is:

1. Set up and activate a conda environment with the Python version and system libraries needed by `gwsurrogate`.
2. Clone the repository on the HPC filesystem:

   ```sh
   git clone git@github.com:sxs-collaboration/gwsurrogate.git
   cd gwsurrogate
   ```

3. Install the package in that environment:

   ```sh
   python -m pip install .
   ```

4. Start an interactive compute node through the scheduler. Once the job starts, record the compute node hostname:

   ```sh
   hostname
   ```

5. From the Anvil login node, connect to that compute node with SSH agent forwarding enabled:

   ```sh
   ssh -A HOSTNAME
   ```

   This forwards your SSH login agent so the benchmark script can fetch a comparison ref from GitHub, such as a pull request branch.

6. For stable HPC timings, pin common BLAS and thread-pool libraries to one
   thread before launching Python:

   ```sh
   export OMP_NUM_THREADS=1
   export OPENBLAS_NUM_THREADS=1
   export MKL_NUM_THREADS=1
   export BLIS_NUM_THREADS=1
   export NUMEXPR_NUM_THREADS=1
   ```

   These variables should be set before running the benchmark so NumPy and the
   BLAS backend see them during Python startup.

7. From the repository checkout on the compute node, run the benchmark. For example, to compare the current checkout with pull request 70:

   ```sh
   python test/benchmark_surrogate_evaluations.py \
     --fetch-ref pull/70/head \
     --compare-ref FETCH_HEAD \
     --compare-label pr-70 \
     --md-output "$PWD/benchmark_surrogate_evaluations.md" \
     --png-output "$PWD/benchmark_surrogate_evaluations.png" \
     --html-output "$PWD/benchmark_surrogate_evaluations.html"
   ```

8. Inspect the generated output files on the HPC filesystem:

   ```sh
   ls test/benchmark_surrogate_evaluations.*
   ```
