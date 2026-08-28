import unittest
from pathlib import Path

from threshold_w2sg.config import (
    CaseConfig,
    ExperimentConfig,
    ModelConfig,
    SolverConfig,
    StateEvolutionConfig,
)
from threshold_w2sg.simulation import run_replicate


class SimulationTests(unittest.TestCase):
    def test_registered_streams_are_exactly_reproducible(self):
        case = CaseConfig("test", "Test", 1.25, 1.40)
        config = ExperimentConfig(
            profile_name="test",
            output_dir=Path("results/test"),
            base_seed=1234,
            p_values=(16,),
            replicates=2,
            bootstrap_resamples=100,
            workers=1,
            model=ModelConfig(0.75, 0.5, 0.05, 0.20),
            cases=(case,),
            solver=SolverConfig(1.0e-8, 20_000, 10, 80, 1.05),
            state_evolution=StateEvolutionConfig(32, 1.0e-10, 3000, 0.6),
            source_path=Path("test.yaml"),
        )
        first = run_replicate(config, case, 0, 16, 1)
        second = run_replicate(config, case, 0, 16, 1)
        for key in [
            "risk1",
            "risk2",
            "risk_difference",
            "stage1_kkt",
            "stage2_kkt",
        ]:
            self.assertEqual(first[key], second[key])


if __name__ == "__main__":
    unittest.main()
