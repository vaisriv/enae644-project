"""Integration tests for simulation controller."""

import pytest
from src.simulation.controller import run_simulation, SimulationResult


class TestRunSimulation:
    """Tests for run_simulation function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_simulation_initialization(self, minimal_simulation_config, jax_key):
        """Test simulation can be initialized from config."""
        # This test will check that run_simulation can at least start
        # even if it hits NotImplementedError in stubs
        with pytest.raises(NotImplementedError):
            result = run_simulation(minimal_simulation_config, jax_key)

    @pytest.mark.skip(reason="Not implemented yet")
    def test_simulation_returns_result(self, minimal_simulation_config, jax_key):
        """Test simulation returns SimulationResult."""
        result = run_simulation(minimal_simulation_config, jax_key)
        assert isinstance(result, SimulationResult)
        assert result.winner in ["Agent_D", "Agent_I", "timeout"]

    @pytest.mark.skip(reason="Not implemented yet")
    def test_agent_d_wins_scenario(self, jax_key):
        """Test scenario where Agent D reaches goal first."""
        from tests.fixtures.sample_configs import config_agent_d_wins
        config = config_agent_d_wins()
        result = run_simulation(config, jax_key)
        assert result.winner == "Agent_D"

    @pytest.mark.skip(reason="Not implemented yet")
    def test_agent_i_wins_scenario(self, jax_key):
        """Test scenario where Agent I intercepts."""
        from tests.fixtures.sample_configs import config_agent_i_wins
        config = config_agent_i_wins()
        result = run_simulation(config, jax_key)
        assert result.winner == "Agent_I"

    @pytest.mark.skip(reason="Not implemented yet")
    def test_timeout_scenario(self, jax_key):
        """Test scenario that times out."""
        from tests.fixtures.sample_configs import config_timeout
        config = config_timeout()
        result = run_simulation(config, jax_key)
        assert result.winner == "timeout"


class TestSimulationResult:
    """Tests for SimulationResult dataclass."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_result_structure(self):
        """Test SimulationResult has required fields."""
        # This will be tested once controller is implemented
        pass
