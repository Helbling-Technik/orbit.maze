"""Launch Isaac Sim Simulator first."""

# from omni.isaac.lab.app import AppLauncher
from isaacsim import SimulationApp

simulation_app = SimulationApp({"hide_ui": False})
from omni.isaac.core.utils.extensions import enable_extension

enable_extension("omni.kit.streamsdk.plugins-3.2.1")
enable_extension("omni.kit.livestream.core-3.2.0")
enable_extension("omni.kit.livestream.native")

"""Rest everything follows."""

import contextlib
import os

import carb
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

import omni.usd
from pxr import UsdShade, Sdf, Gf
from omni.physx.scripts import utils
import omni.isaac.core.utils.stage as stage_utils


def main():
    usd_path = os.path.abspath("urdfs/generated_maze_03/generated_maze_03.usd")

    usd_context = omni.usd.get_context()
    stage_utils.open_stage(usd_path)
    # result, error_str = usd_context.open_stage_async(usd_path)

    stage = usd_context.get_stage()

    robot = stage.GetPrimAtPath("/Labyrinth")

    walls = stage.GetPrimAtPath("/Labyrinth/InnerDOFWalls")
    supportvisual = stage.GetPrimAtPath("/Labyrinth/Support/visuals")
    outerDOFvisual = stage.GetPrimAtPath("/Labyrinth/OuterDOF/visuals")
    innerDOFvisual = stage.GetPrimAtPath("/Labyrinth/InnerDOF/visuals")
    innerDOFWallsvisual = stage.GetPrimAtPath("/Labyrinth/InnerDOFWalls/visuals")
    utils.setCollider(supportvisual, "sdf")
    utils.setCollider(outerDOFvisual, "sdf")
    utils.setCollider(innerDOFvisual, "sdf")
    utils.setCollider(innerDOFWallsvisual, "sdf")

    mtl_created_list = []
    omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary",
        mdl_name="OmniGlass.mdl",
        mtl_name="OmniGlass",
        mtl_created_list=mtl_created_list,
    )

    mtl_prim = stage.GetPrimAtPath(mtl_created_list[0])
    omni.usd.create_material_input(mtl_prim, "glass_color", Gf.Vec3f(0, 1, 0), Sdf.ValueTypeNames.Color3f)
    omni.usd.create_material_input(mtl_prim, "glass_ior", 1.0, Sdf.ValueTypeNames.Float)

    cube_mat_shade = UsdShade.Material(mtl_prim)
    UsdShade.MaterialBindingAPI(walls).Bind(cube_mat_shade, UsdShade.Tokens.strongerThanDescendants)

    # PhysicsSchemaTools.addGroundPlane(stage, "/ground_plane", 100.0, 100.0, 0.0)
    # Reinitialize the simulation
    app = omni.kit.app.get_app_interface()
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
