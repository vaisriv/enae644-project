# Adversarial Motion Planning

ENAE644 Term Project implementing two-agent adversarial planning in continuous 2D space.

## Overview

Two competing agents with learning-based and game-theoretic planning:

- **Deceptive Agent**: Uses Adversarial RRT\* with learned deception costs to conceal its true objective
- **Interceptor Agent**: Employs inverse reinforcement learning and game-theoretic prediction to infer goals and plan interception

Built with [Python 3.13](https://python.org/downloads/release/python-31311) ([JAX](https://docs.jax.dev/en/latest)/[Equinox](https://docs.kidger.site/equinox)), [Typst](https://typst.app/home), and [Nix](https://nixos.org).

## Quick Start

This project uses [Nix Flakes](https://nix.dev/concepts/flakes.html) for reproducible packaging and development:

```bash
# Enter development environment
nix develop

# Run simulation
nix run .#adversarial-planning

# Compile the report
nix run .#report.build

# Or watch and continuously compile the report
nix run .#report.watch
```

## Common Commands

```bash
# Helper for running the simulation
py                      # Executes Python program

# Helper for writing the report
typ                     # Watch and compile Typst report

# Format code
nix fmt                 # Format all files (Python, Nix, Typst, YAML, Markdown)
```

## Project Structure

```
.
├─ src/                 # Python implementation
│  ├── index.py         # Main entrypoint (symlinked as submission.py)
│  ├── deceptive/       # Deceptive agent (Adversarial RRT*)
│  ├── interceptor/     # Interceptor agent (IRL + game-theoretic planning)
│  └── simulation/      # Simulation environment
│
├─ reports/main.typ     # Research report (IEEE format)
├─ outputs/             # Generated figures and data
├─ references/          # Papers and assignment PDFs
│  ├── assignments/     # Progress report requirements
│  └── papers/          # PDFs of cited papers/works
└── docs/spec/          # Detailed software specifications
```

## Implementation Stack

- **Python**: JAX for automatic differentiation and GPU acceleration, Equinox for neural networks
- **Report**: Typst (IEEE conference format)
- **Environment**: Nix Flakes for reproducible builds
