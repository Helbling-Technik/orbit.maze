# Extension Template for Orbit

[![IsaacSim](https://img.shields.io/badge/IsaacSim-2023.1.1-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Orbit](https://img.shields.io/badge/Orbit-0.2.0-silver)](https://isaac-orbit.github.io/orbit/)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

## Overview

This repository serves as a template for building projects or extensions based on Orbit. It allows you to develop in an isolated environment, outside of the core Orbit repository. Furthermore, this template serves three use cases:

- **Python Package**
    Can be installed into Isaac Sim's Python environment, making it suitable for users who want to integrate their extension to `Orbit` as a python package.

- **Project Template**
    Ensures access to `Isaac Sim` and `Orbit` functionalities, which can be used as a project template.

- **Omniverse Extension**
    Can be used as an Omniverse extension, ideal for projects that leverage the Omniverse platform's graphical user interface.

**Key Features:**

- `Isolation` Work outside the core Orbit repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, orbit

### License

The source code is released under a [BSD 3-Clause license](https://opensource.org/licenses/BSD-3-Clause).

**Author: The ORBIT Project Developers<br />
Affiliation: [The AI Institute](https://theaiinstitute.com/)<br />
Maintainer: Nico Burger, nburger@theaiinstitute.com**

## Setup

Depending on the use case defined [above](#overview), follow the instructions to set up your extension template. Start with the [Basic Setup](#basic-setup), which is required for either use case.

### Basic Setup

#### Dependencies

This template depends on Isaac Sim and Orbit. For detailed instructions on how to install these dependencies, please refer to the [installation guide](https://isaac-orbit.github.io/orbit/source/setup/installation.html).

- [Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html)
- [Orbit](https://isaac-orbit.github.io/orbit/)

#### Configuration

- Set up a symbolic link from Orbit to this directory.
This makes it convenient to index the python modules and look for extensions shipped with Isaac Sim and Orbit.

```bash
ln -s <your_orbit_path> _orbit
```

#### Environment (Optional)

For clarity, we will be using the `${ISAACSIM_PATH}/python.sh` command to call the Orbit specific python interpreter. However, you might be working from within a virtual environment, allowing you to use the `python` command directly, instead of `${ISAACSIM_PATH}/python.sh`. Information on setting up a virtual environment for Orbit can be found [here](https://isaac-orbit.github.io/orbit/source/setup/installation.html#setting-up-the-environment). The `ISAACSIM_PATH` should already be set from installing Orbit, see [here](https://isaac-orbit.github.io/orbit/source/setup/installation.html#configuring-the-environment-variables).

#### Configure Python Interpreter

In the provided configuration, we set the default Python interpreter to use the Python executable provided by Omniverse. This is specified in the `.vscode/settings.json` file:

```json
"python.defaultInterpreterPath": "${env:ISAACSIM_PATH}/python.sh"
```

This setup requires you to have set up the `ISAACSIM_PATH` environment variable. If you want to use a different Python interpreter, you need to change the Python interpreter used by selecting and activating the Python interpreter of your choice in the bottom left corner of VSCode, or opening the command palette (`Ctrl+Shift+P`) and selecting `Python: Select Interpreter`.

#### Set up IDE

To setup the IDE, please follow these instructions:

1. Open the `orbit.maze` directory on Visual Studio Code IDE
2. Run VSCode Tasks, by pressing Ctrl+Shift+P, selecting Tasks: Run Task and running the setup_python_env in the drop down menu.

If everything executes correctly, it should create a file .python.env in the .vscode directory. The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse. This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Python Package / Project Template

From within this repository, install your extension as a Python package to the Isaac Sim Python executable.

```bash
${ISAACSIM_PATH}/python.sh -m pip install --upgrade pip
${ISAACSIM_PATH}/python.sh -m pip install -e .
```

### Setup as Omniverse Extension

To enable your extension, follow these steps:

1. **Add the search path of your repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon** (☰), then go to `Settings`.
    - In the `Extension Search Paths`, enter the path that goes up to your repository's location without actually including the repository's own directory. For example, if your repository is located at `/home/code/orbit.ext_template`, you should add `/home/code` as the search path.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to your local Orbit directory. For example: `/home/orbit/source/extensions`
    - Click on the **Hamburger Icon** (☰), then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Usage

### Python Package

Import your python package within `Isaac Sim` and `Orbit` using:

```python
import orbit.<your_extension_name>
```

### Project Template

We provide an example for training and playing a policy for ANYmal on flat terrain. Install [RSL_RL](https://github.com/leggedrobotics/rsl_rl) outside of the orbit repository, e.g. `home/code/rsl_rl`.

```bash
git clone https://github.com/leggedrobotics/rsl_rl.git
cd rsl_rl
${ISAACSIM_PATH}/python.sh -m pip install -e .
```

Train a policy.

```bash
cd <path_to_your_extension>
${ISAACSIM_PATH}/python.sh scripts/sb3/train.py --task Isaac-Maze-v0 --num_envs 4096 --headless
```

Play the trained policy.

```bash
${ISAACSIM_PATH}/python.sh scripts/sb3/play.py --task Isaac-Maze-v0 --num_envs 16
```

## Bugs & Feature Requests

Please report bugs and request features using the [Issue Tracker](https://github.com/isaac-orbit/orbit.ext_template/issues).
