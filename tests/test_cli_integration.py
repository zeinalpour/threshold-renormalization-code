import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd


class CliIntegrationTests(unittest.TestCase):
    def test_run_render_and_recompute_state_evolution(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "results"
            config = root / "integration.yaml"
            config.write_text(
                f"""profile_name: integration
output_dir: {output.as_posix()}
base_seed: 1701
p_values: [12]
replicates: 2
bootstrap_resamples: 100
workers: 1
model:
  gamma: 0.75
  delta: 0.5
  nu1: 0.05
  nu2: 0.20
cases:
  - {{id: integration_case, label: Integration case, alpha1: 1.25, alpha2: 1.40}}
solver:
  kkt_tolerance: 1.0e-7
  max_iterations: 10000
  check_every: 10
  power_iterations: 80
  spectral_safety: 1.05
state_evolution:
  quadrature_nodes: 128
  tolerance: 1.0e-9
  max_iterations: 2000
  damping: 0.60
""",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            source = str(project / "src")
            environment["PYTHONPATH"] = source + os.pathsep + environment.get(
                "PYTHONPATH", ""
            )

            def invoke(command: str) -> None:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "threshold_w2sg.cli",
                        command,
                        "--config",
                        str(config),
                    ],
                    cwd=project,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    msg=completed.stdout + "\n" + completed.stderr,
                )

            invoke("run")
            invoke("render")
            invoke("recompute-se")
            empirical = pd.read_csv(output / "raw" / "empirical_replicates.csv")
            state = pd.read_csv(output / "raw" / "state_evolution.csv")
            metadata = json.loads(
                (output / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(empirical), 2)
            self.assertEqual(len(state), 1)
            self.assertGreater(float(state.loc[0, "active1"]), 0.0)
            self.assertLess(float(state.loc[0, "active1"]), 1.0)
            self.assertGreater(float(state.loc[0, "active2"]), 0.0)
            self.assertLess(float(state.loc[0, "active2"]), 1.0)
            self.assertIn("analytic Stage-I", metadata["state_evolution_method"])
            self.assertTrue((output / "figures" / "phase_diagram.pdf").exists())
            self.assertTrue(
                (output / "latex" / "numerical_validation_generated.tex").exists()
            )


if __name__ == "__main__":
    unittest.main()
