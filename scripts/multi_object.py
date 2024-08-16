# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn multiple objects in multiple environments.

.. code-block:: bash

    # Usage
    python scripts/multi_object.py --livestream 1 --num_envs 2

"""
from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Demo on spawning different objects in multiple environments.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import yaml
import traceback
from dataclasses import MISSING

import carb
import omni.isaac.core.utils.prims as prim_utils
from pxr import Usd
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import Timer, configclass
from omni.isaac.lab.actuators import ImplicitActuatorCfg
import omni.isaac.core.utils.stage as stage_utils

from omni.isaac.lab.sim import schemas
from omni.isaac.lab.sim.utils import bind_visual_material, select_usd_variants

import math
import re


# This is from isaac lab directly, but not includable without modifying isaacs code
def _spawn_from_usd_file(
    prim_path: str,
    usd_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn an asset from a USD file and override the settings with the given config.

    In case a prim already exists at the given prim path, then the function does not create a new prim
    or throw an error that the prim already exists. Instead, it just takes the existing prim and overrides
    the settings with the given config.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        usd_path: The path to the USD file to spawn the asset from.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
            case the translation specified in the generated USD file is used.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the orientation specified in the generated USD file is used.

    Returns:
        The prim of the spawned asset.

    Raises:
        FileNotFoundError: If the USD file does not exist at the given path.
    """
    # check file path exists
    stage: Usd.Stage = stage_utils.get_current_stage()
    if not stage.ResolveIdentifierToEditTarget(usd_path):
        raise FileNotFoundError(f"USD file not found at path: '{usd_path}'.")
    # spawn asset if it doesn't exist.
    if not prim_utils.is_prim_path_valid(prim_path):
        # add prim as reference to stage
        prim_utils.create_prim(
            prim_path,
            usd_path=usd_path,
            translation=translation,
            orientation=orientation,
            scale=cfg.scale,
        )
    else:
        carb.log_warn(f"A prim already exists at prim path: '{prim_path}'.")

    # modify variants
    if hasattr(cfg, "variants") and cfg.variants is not None:
        select_usd_variants(prim_path, cfg.variants)

    # modify rigid body properties
    if cfg.rigid_props is not None:
        schemas.modify_rigid_body_properties(prim_path, cfg.rigid_props)
    # modify collision properties
    if cfg.collision_props is not None:
        schemas.modify_collision_properties(prim_path, cfg.collision_props)
    # modify mass properties
    if cfg.mass_props is not None:
        schemas.modify_mass_properties(prim_path, cfg.mass_props)

    # modify articulation root properties
    if cfg.articulation_props is not None:
        schemas.modify_articulation_root_properties(prim_path, cfg.articulation_props)
    # modify tendon properties
    if cfg.fixed_tendons_props is not None:
        schemas.modify_fixed_tendon_properties(prim_path, cfg.fixed_tendons_props)
    # define drive API on the joints
    # note: these are only for setting low-level simulation properties. all others should be set or are
    #  and overridden by the articulation/actuator properties.
    if cfg.joint_drive_props is not None:
        schemas.modify_joint_drive_properties(prim_path, cfg.joint_drive_props)

    # modify deformable body properties
    if cfg.deformable_props is not None:
        schemas.modify_deformable_body_properties(prim_path, cfg.deformable_props)

    # apply visual material
    if cfg.visual_material is not None:
        if not cfg.visual_material_path.startswith("/"):
            material_path = f"{prim_path}/{cfg.visual_material_path}"
        else:
            material_path = cfg.visual_material_path
        # create material
        cfg.visual_material.func(material_path, cfg.visual_material)
        # apply material
        bind_visual_material(prim_path, material_path)

    # return the prim
    return prim_utils.get_prim_at_path(prim_path)


def spawn_multi_mazes(
    prim_path: str,
    cfg: MultiMazeCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:

    # return the prim
    # return _spawn_from_usd_file(prim_path, usd_path, cfg, translation, orientation)
    # resolve: {SPAWN_NS}/AssetName
    # note: this assumes that the spawn namespace already exists in the stage
    root_path, asset_path = prim_path.rsplit("/", 1)
    # check if input is a regex expression
    # note: a valid prim path can only contain alphanumeric characters, underscores, and forward slashes
    is_regex_expression = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None

    # resolve matching prims for source prim path expression
    if is_regex_expression and root_path != "":
        source_prim_paths = sim_utils.find_matching_prim_paths(root_path)
        # if no matching prims are found, raise an error
        if len(source_prim_paths) == 0:
            raise RuntimeError(
                f"Unable to find source prim path: '{root_path}'. Please create the prim before spawning."
            )
    else:
        source_prim_paths = [root_path]

    # resolve prim paths for spawning
    prim_paths = [f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths]
    # spawn asset from the given usd file
    for idx, prim_path in enumerate(prim_paths):
        # sample the asset config to load
        usd_idx = idx % len(cfg.maze_usd_cfgs)
        usd_path = os.path.join(cfg.project_root, cfg.maze_usd_cfgs[usd_idx].usd_path)
        usd_cfg = cfg.maze_usd_cfgs[usd_idx]

        print(f"Prim path {prim_path}")
        print(f"USD path {usd_path}")
        # load the asset
        prim = _spawn_from_usd_file(prim_path, usd_path, usd_cfg, translation, orientation)

    return prim


def get_maze_cfg():
    # articulation
    # load maze path from yaml file
    yaml_path = "usds/multi_usd_paths.yaml"
    with open(os.path.join(yaml_path), "r") as file:
        data = yaml.safe_load(file)
        maze_usd_paths = data["usd_paths"]

    usd_file_cfgs = []
    for usd in maze_usd_paths:
        maze_usd_cfg = sim_utils.UsdFileCfg(
            usd_path=usd,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
                enable_gyroscopic_forces=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            # physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.2, dynamic_friction=0.2),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        )
        usd_file_cfgs.append(maze_usd_cfg)

    maze_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Labyrinth",
        spawn=MultiMazeCfg(maze_usd_cfgs=usd_file_cfgs),
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

    return maze_cfg


@configclass
class MultiMazeCfg(sim_utils.SpawnerCfg):
    """Configuration parameters for loading multiple mazes looping over a list"""

    func: sim_utils.SpawnerCfg.func = spawn_multi_mazes
    """Overriden spawner function"""
    maze_usd_cfgs: list[sim_utils.UsdFileCfg] = MISSING
    """List of mazes to spawn, usd configs."""
    current_script_path = os.path.abspath(__file__)
    # Absolute path of the project root (assuming it's two levels up from the current script)
    project_root = os.path.join(current_script_path, "../..")


@configclass
class MultiObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a multi-object scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    maze_cfg = get_maze_cfg()

    # Sphere with collision enabled but not actuated
    sphere = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.00625,
            mass_props=sim_utils.MassPropertiesCfg(density=7850),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2), metallic=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.2, dynamic_friction=0.2),
        ),
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(globals.maze_path[0, 0], globals.maze_path[0, 1], 0.12)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.12)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),
    )


# Just for testing purposes, not needed for final implementation
def reset_scene_to_default(scene: InteractiveScene):
    """Reset the scene to the default state specified in the scene configuration."""
    # rigid bodies
    for rigid_object in scene.rigid_objects.values():
        # obtain default and deal with the offset for env origins
        default_root_state = rigid_object.data.default_root_state.clone()
        default_root_state[:, 0:3] += scene.env_origins
        # set into the physics simulation
        rigid_object.write_root_state_to_sim(default_root_state)
    # articulations
    for articulation_asset in scene.articulations.values():
        # obtain default and deal with the offset for env origins
        default_root_state = articulation_asset.data.default_root_state.clone()
        default_root_state[:, 0:3] += scene.env_origins
        # set into the physics simulation
        articulation_asset.write_root_state_to_sim(default_root_state)
        # obtain default joint positions
        default_joint_pos = articulation_asset.data.default_joint_pos.clone()
        default_joint_vel = articulation_asset.data.default_joint_vel.clone()
        # set into the physics simulation
        articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel)


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            reset_scene_to_default(scene)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device="cpu", use_gpu_pipeline=False)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    # sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = MultiObjectSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.5, replicate_physics=False)
    with Timer("[INFO] Time to create scene: "):
        scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
