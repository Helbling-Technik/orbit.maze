# Extension Template for IsaacLab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-2023.1.1-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Orbit](https://img.shields.io/badge/Orbit-0.2.0-silver)](https://isaac-orbit.github.io/orbit/)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)


## Setup

Depending on the use case defined [above](#overview), follow the instructions to set up your extension template. Start with the [Basic Setup](#basic-setup), which is required for either use case.

### Basic Setup

#### Dependencies

This template depends on Isaac Sim and Orbit. For detailed instructions on how to install these dependencies, please refer to the [installation guide](https://isaac-orbit.github.io/orbit/source/setup/installation.html).

- [Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)

The orbit maze repo works with an older orbit version before it got renamed as Isaac lab. The commit hash of the older Orbit version is 1928161b775abb2ccde112c2246653e70253a560.
The Isaac Sim version is the 2023.1.1 release.

Update: This package is now fully ported to track the new mainline Isaac Lab and works with Isaac-Sim 4.1.

#### Configuration

- Set up a symbolic link from Isaac-Lab to this directory.
This makes it convenient to index the python modules and look for extensions shipped with Isaac Sim and Orbit.

```bash
ln -s <your_isaaclab_path> _isaaclab
```

#### Environment (Optional)

For clarity, we will be using the `${ISAACSIM_PATH}/python.sh` command to call the Orbit specific python interpreter. However, you might be working from within a virtual environment, allowing you to use the `python` command directly, instead of `${ISAACSIM_PATH}/python.sh`. 

Information on setting up a virtual environment for IsaacLab can be found [here](https://isaac-sim.github.io/IsaacLab/source/setup/installation/binaries_installation.html#installing-isaac-lab). The `ISAACSIM_PATH` should already be set from installing IsaacLab. 

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
## Usage

### Python Package

Import your python package within `Isaac Sim` and `IsaacLab` using:

```python
import orbit.<your_extension_name>
```

### Project Template

Train a policy.

```bash
cd <path_to_your_extension>
${ISAACSIM_PATH}/python.sh scripts/sb3/train.py --task Isaac-Maze-v0 --num_envs 4096 --headless
```

Play the trained policy.

```bash
${ISAACSIM_PATH}/python.sh scripts/sb3/play.py --task Isaac-Maze-v0 --num_envs 16
```

make sure to activate your virtual environment first. Sometimes one needs to run the command twice.

```bash
conda activate issaclab
```

### Remote Livestreaming

In order to be able to livestream the simulator to another machine one needs to change the default version of the extensions in the ``AppLauncher``
`` /home/sck/git/IsaacLab/source/extensions/omni.isaac.lab/omni/isaac/lab/app/app_launcher.py ``

Exchange lines 567 to 569 with 
``` python
enable_extension("omni.kit.streamsdk.plugins-3.2.1")
enable_extension("omni.kit.livestream.core-3.2.0")
enable_extension("omni.kit.livestream.native")
```
