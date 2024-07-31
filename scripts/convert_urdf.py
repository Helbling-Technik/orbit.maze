# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to convert a URDF into USD format.

Unified Robot Description Format (URDF) is an XML file format used in ROS to describe all elements of
a robot. For more information, see: http://wiki.ros.org/urdf

This script uses the URDF importer extension from Isaac Sim (``omni.isaac.urdf_importer``) to convert a
URDF asset into USD format. It is designed as a convenience script for command-line use. For more
information on the URDF importer, see the documentation for the extension:
https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/ext_omni_isaac_urdf.html


positional arguments:
  input               The path to the input URDF file.
  output              The path to store the USD file.

optional arguments:
  -h, --help                Show this help message and exit
  --merge-joints            Consolidate links that are connected by fixed joints. (default: False)
  --fix-base                Fix the base to where it is imported. (default: False)
  --make-instanceable       Make the asset instanceable for efficient cloning. (default: False)

"""

"""Launch Isaac Sim Simulator first."""

import argparse

# from omni.isaac.lab.app import AppLauncher
from isaacsim import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to convert a URDF into USD format.")
parser.add_argument(
    "--input",
    type=str,
    default="urdfs/generated_maze_03/generated_maze_03.urdf",
    help="The path to the input URDF file.",
)
parser.add_argument(
    "--output",
    type=str,
    default="urdfs/generated_maze_03/generated_maze_03",
    help="The path to store the USD file.",
)
parser.add_argument(
    "--merge-joints",
    action="store_true",
    default=False,
    help="Consolidate links that are connected by fixed joints.",
)
parser.add_argument("--fix-base", action="store_true", default=True, help="Fix the base to where it is imported.")
parser.add_argument(
    "--make-instanceable",
    action="store_true",
    default=True,
    help="Make the asset instanceable for efficient cloning.",
)

# # parse the arguments
args_cli = parser.parse_args()

simulation_app = SimulationApp({"hide_ui": False})
from omni.isaac.core.utils.extensions import enable_extension

enable_extension("omni.kit.streamsdk.plugins-3.2.1")
enable_extension("omni.kit.livestream.core-3.2.0")
enable_extension("omni.kit.livestream.native")

"""Rest everything follows."""

import contextlib
import os

import carb
import omni.isaac.core.utils.stage as stage_utils
import omni.kit.app

from omni.isaac.lab.sim.converters import UrdfConverter, UrdfConverterCfg
from omni.isaac.lab.utils.assets import check_file_path
from omni.isaac.lab.utils.dict import print_dict

from omni.isaac.kit import SimulationApp
from omni.isaac.core import World
from omni.isaac.core.utils import prims

from omni.isaac.lab.sim.spawners.lights import LightCfg, spawn_light
from omni.isaac.lab.assets import ArticulationCfg, Articulation
from omni.isaac.lab.sim.schemas import ArticulationRootPropertiesCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
import omni.isaac.lab.sim as sim_utils
import omni.physics.tensors.impl.api as physx
import math
import torch


def main():
    # check valid file path
    urdf_path = args_cli.input
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.abspath(urdf_path)
    if not check_file_path(urdf_path):
        raise ValueError(f"Invalid file path: {urdf_path}")
    # create destination path
    dest_path = args_cli.output
    if not os.path.isabs(dest_path):
        dest_path = os.path.abspath(dest_path)

    # Create Urdf converter config
    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        fix_base=args_cli.fix_base,
        merge_fixed_joints=args_cli.merge_joints,
        force_usd_conversion=True,
        make_instanceable=args_cli.make_instanceable,
    )

    # Print info
    print("-" * 80)
    print("-" * 80)
    print(f"Input URDF file: {urdf_path}")
    print("URDF importer config:")
    print_dict(urdf_converter_cfg.to_dict(), nesting=0)
    print("-" * 80)
    print("-" * 80)

    # Create Urdf converter and import the file
    urdf_converter = UrdfConverter(urdf_converter_cfg)
    # print output
    print("URDF importer output:")
    print(f"Generated USD file: {urdf_converter.usd_path}")
    print("-" * 80)
    print("-" * 80)

    # Determine if there is a GUI to update:
    # acquire settings interface
    carb_settings_iface = carb.settings.get_settings()
    # read flag for whether a local GUI is enabled
    local_gui = carb_settings_iface.get("/app/window/enabled")
    # read flag for whether livestreaming GUI is enabled
    livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

    # Simulate scene (if not headless)
    if local_gui or livestream_gui:
        # Reinitialize the simulation
        app = omni.kit.app.get_app_interface()

        # create prim
        world = World(stage_units_in_meters=1.0)

        robot_cfg = ArticulationCfg(
            prim_path="/Labyrinth",
            spawn=sim_utils.UsdFileCfg(
                usd_path=urdf_converter.usd_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=100.0,
                    enable_gyroscopic_forces=True,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.001,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0), joint_pos={"OuterDOF_RevoluteJoint": 0.0, "InnerDOF_RevoluteJoint": 0.0}
            ),
            # Position Control: For position controlled joints, set a high stiffness and relatively low or zero damping.
            # Velocity Control: For velocity controller joints, set a high damping and zero stiffness.
            actuators={
                "outer_actuator": ImplicitActuatorCfg(
                    joint_names_expr=["OuterDOF_RevoluteJoint"],
                    effort_limit=10,  # 5g * 9.81 * 0.15m = 0.007357
                    velocity_limit=20 * math.pi,
                    stiffness=0.0,
                    damping=10.0,
                ),
                "inner_actuator": ImplicitActuatorCfg(
                    joint_names_expr=["InnerDOF_RevoluteJoint"],
                    effort_limit=10,  # 5g * 9.81 * 0.15m = 0.007357
                    velocity_limit=20 * math.pi,
                    stiffness=0.0,
                    damping=10.0,
                ),
            },
        )

        robot = Articulation(robot_cfg)
        # joint_ids = torch.tensor([0, 1], dtype=torch.int, device="cuda:0")
        # print(robot.data())
        # robot.write_joint_limits_to_sim(limits=15 / 180 * math.pi, joint_ids=joint_ids)
        limits = torch.tensor([15 / 180 * math.pi, 15 / 180 * math.pi], dtype=torch.float32, device="cuda:0")
        indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda:0")
        ArtView = physx.ArticulationView()
        ArtView.set_dof_limits(data=limits, indices=indices)

        art_cfg = ArticulationRootPropertiesCfg(articulation_enabled=True, fix_root_link=True)
        sim_utils.schemas.modify_articulation_root_properties("/Labyrinth", art_cfg)

        # robot = prims.create_prim("/Labyrinth", usd_path=urdf_converter.usd_path)
        # prims.create_prim("/DomeLight", "DomeLight")
        light_cfg = LightCfg(intensity=1000.0, prim_type="DomeLight")
        light = spawn_light("/DomeLight", light_cfg)

        # Run simulation
        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():
                # perform step
                app.update()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
