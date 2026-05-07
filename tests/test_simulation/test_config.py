"""Tests for configuration loading and validation."""

import dataclasses

import pytest
from src.simulation.config import (
    SimulationConfig,
    load_config,
    _validate_config,
    config_to_dict,
    dict_to_config,
)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self, sample_config_yaml):
        """Test loading a valid YAML configuration file."""
        config = load_config(sample_config_yaml)
        assert isinstance(config, SimulationConfig)
        assert config.simulation_params.random_seed == 42

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


class TestValidateConfig:
    """Tests for _validate_config function."""

    def test_validate_valid_config(self, minimal_simulation_config):
        """Test validation passes for valid config."""
        _validate_config(minimal_simulation_config)  # should not raise

    def test_validate_true_goal_not_in_candidates(self, minimal_simulation_config):
        """Test validation fails when true goal not in candidate goals."""
        bad_config = dataclasses.replace(
            minimal_simulation_config,
            deceptive_agent_config=dataclasses.replace(
                minimal_simulation_config.deceptive_agent_config,
                true_goal=[99.0, 99.0],
            ),
        )
        with pytest.raises(ValueError):
            _validate_config(bad_config)

    def test_validate_deception_weight_out_of_range(self, minimal_simulation_config):
        """Test validation fails for deception weight outside [0, 1]."""
        bad_planner = dataclasses.replace(
            minimal_simulation_config.deceptive_agent_config.planner,
            deception_weight=1.5,
        )
        bad_config = dataclasses.replace(
            minimal_simulation_config,
            deceptive_agent_config=dataclasses.replace(
                minimal_simulation_config.deceptive_agent_config,
                planner=bad_planner,
            ),
        )
        with pytest.raises(ValueError):
            _validate_config(bad_config)

    def test_validate_candidate_goals_mismatch(self, minimal_simulation_config):
        """Test validation fails when agent candidate goals don't match."""
        bad_config = dataclasses.replace(
            minimal_simulation_config,
            interceptor_agent_config=dataclasses.replace(
                minimal_simulation_config.interceptor_agent_config,
                candidate_goals=[[1.0, 1.0]],
            ),
        )
        with pytest.raises(ValueError):
            _validate_config(bad_config)


class TestConfigSerialization:
    """Tests for config serialization/deserialization."""

    def test_config_to_dict(self, minimal_simulation_config):
        """Test converting config to dictionary uses YAML-schema keys."""
        config_dict = config_to_dict(minimal_simulation_config)
        assert isinstance(config_dict, dict)
        assert "workspace" in config_dict
        assert "deceptive_agent" in config_dict
        assert "interceptor_agent" in config_dict
        assert "simulation" in config_dict
        assert "training" in config_dict

    def test_dict_to_config_roundtrip(self, minimal_simulation_config):
        """Test converting config to dict and back preserves values."""
        config_dict = config_to_dict(minimal_simulation_config)
        restored_config = dict_to_config(config_dict)
        assert (
            restored_config.simulation_params.random_seed
            == minimal_simulation_config.simulation_params.random_seed
        )
        assert (
            restored_config.deceptive_agent_config.true_goal
            == minimal_simulation_config.deceptive_agent_config.true_goal
        )
        assert (
            restored_config.deceptive_agent_config.candidate_goals
            == minimal_simulation_config.deceptive_agent_config.candidate_goals
        )
