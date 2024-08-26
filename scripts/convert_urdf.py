# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

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
    default="urdfs/converter_input/generated/generated_maze_rov.urdf",
    help="The path to the input URDF file.",
)
parser.add_argument(
    "--output",
    type=str,
    default="urdfs/converter_output/generated_maze_rov_02_jointLimit",
    help="The path to store the USD file.",
)
parser.add_argument(
    "--run_gui",
    action="store_true",
    default=True,
    help="Run simulation gui to verify visually if convert was correct",
)
parser.add_argument("--fix_base", action="store_true", default=True, help="Fix the base to where it is imported.")
parser.add_argument(
    "--make_instanceable",
    action="store_true",
    default=False,
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

import omni.kit.app

from omni.isaac.lab.sim.converters import UrdfConverter, UrdfConverterCfg
from omni.isaac.lab.utils.assets import check_file_path
from omni.isaac.lab.utils.dict import print_dict

import omni.usd
from pxr import UsdPhysics, UsdShade, Sdf, Gf
from omni.physx.scripts import utils
import omni.isaac.core.utils.stage as stage_utils


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
    # TODO we would like to have it instanceable but then we have to update the references since they are wrong and change material in reference?
    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        fix_base=args_cli.fix_base,
        merge_fixed_joints=False,
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

    # Now to adjust the usd parameters
    usd_path = os.path.abspath(urdf_converter.usd_path)

    usd_context = omni.usd.get_context()
    stage_utils.open_stage(usd_path)
    stage = usd_context.get_stage()

    robot = stage.GetPrimAtPath("/Labyrinth")
    supportvisual = stage.GetPrimAtPath("/Labyrinth/Support/visuals")
    outerDOFvisual = stage.GetPrimAtPath("/Labyrinth/OuterDOF/visuals")
    innerDOFvisual = stage.GetPrimAtPath("/Labyrinth/InnerDOF/visuals")
    innerDOFWallsvisual = stage.GetPrimAtPath("/Labyrinth/InnerDOFWalls/visuals")

    # Define the path for the physics scene
    physics_scene_path = "/physicsScene"

    # Create the physics scene if it doesn't exist
    if not stage.GetPrimAtPath(physics_scene_path):
        physics_scene = UsdPhysics.Scene.Define(stage, physics_scene_path)

        # Set gravity for the physics scene
        gravity_vector = Gf.Vec3f(0.0, 0.0, -9.81)
        physics_scene.CreateGravityDirectionAttr(gravity_vector)
        physics_scene.CreateGravityMagnitudeAttr(gravity_vector.GetLength())

    # Set the colliders
    utils.setCollider(supportvisual, "sdf")
    utils.setCollider(outerDOFvisual, "sdf")
    utils.setCollider(innerDOFvisual, "sdf")
    utils.setCollider(innerDOFWallsvisual, "sdf")

    # TODO use something like this if you plan on doing instanced meshes
    # Create a new reference
    # new_reference = Sdf.Reference("path/to/new/mesh.usd", "/path/in/mesh.usd")

    # # Clear existing references and set the new one
    # mesh_prim.GetReferences().ClearReferences()
    # mesh_prim.GetReferences().AddReference(new_reference)

    # Reset the mass to autocompute for rigid bodies and ignore diagonal inertia
    for prim in stage.Traverse():
        if prim.GetPath().HasPrefix(robot.GetPath()):
            if UsdPhysics.RigidBodyAPI(prim):
                if UsdPhysics.MassAPI(prim):
                    usd_physics_mass_api = UsdPhysics.MassAPI(prim)

                    mass_attr = usd_physics_mass_api.GetMassAttr()
                    if mass_attr.HasValue():
                        mass_attr.Clear()
                        print(f"Mass for {prim} set to auto (mass attribute cleared).")
                    else:
                        print(f"Mass attribute is already set to auto for {prim}.")

                    inertia_attr = usd_physics_mass_api.GetDiagonalInertiaAttr()
                    if inertia_attr.HasValue():
                        inertia_attr.Clear()
                        print(f"Inertia for {prim} set to ignore.")
                    else:
                        print(f"Inertia is already set to ignore for {prim}.")

    # Define the material path
    material_path = "/Labyrinth/Looks/BlackMaterial"
    # Create a new material
    material = UsdShade.Material.Define(stage, material_path)

    # Create a surface shader
    shader = UsdShade.Shader.Define(stage, material_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")

    # Set the diffuse color to black (RGB values of 0, 0, 0)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))

    # Connect the shader to the material
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # Ensure the mesh exists before applying the material
    if innerDOFWallsvisual:
        UsdShade.MaterialBindingAPI(innerDOFWallsvisual).Bind(material)
    else:
        print(f"Mesh at path {innerDOFWallsvisual} does not exist.")

    # change to proper limits: lr = +-10 InnerJoint; vh = +-7 OuterJoint
    # In revolutejoint set limits and in drive change stiffness 10000000 and damping 100000 and max force 1000
    for prim in stage.Traverse():
        if prim.GetPath().HasPrefix(robot.GetPath()):
            # Ensure the prim is a revolute joint
            if "revolute" in str(prim.GetPath()).lower():
                if "outerdof_revolutejoint" in str(prim.GetPath()).lower():
                    joint_limit = 7.0
                else:
                    joint_limit = 10.0
                # Set revolute joint properties
                if UsdPhysics.RevoluteJoint(prim):
                    # Access the RevoluteJointAPI
                    joint_api = UsdPhysics.RevoluteJoint(prim)

                    lower_attr = joint_api.GetLowerLimitAttr()
                    if lower_attr:
                        lower_attr.Set(-joint_limit)
                        print(f"Lower Limit for {prim.GetPath()} set to -{joint_limit}")
                    else:
                        print(f"No Lower Limit attribute found for {prim.GetPath()}.")

                    upper_attr = joint_api.GetUpperLimitAttr()
                    if upper_attr:
                        upper_attr.Set(joint_limit)
                        print(f"Upper Limit for {prim.GetPath()} set to {joint_limit}")
                    else:
                        print(f"No Upper Limit attribute found for {prim.GetPath()}.")
                else:
                    print(f"No revolute joint found at {prim.GetPath()}.")

                # Set drive properties
                if UsdPhysics.DriveAPI(prim, "angular"):
                    # Access the DriveAPI
                    drive_api = UsdPhysics.DriveAPI(prim, "angular")

                    stiffness_attr = drive_api.GetStiffnessAttr()
                    if stiffness_attr:
                        stiffness_attr.Set(10000000.0)
                        print(f"Stiffness for {prim.GetPath()} set to 10000000.0.")
                    else:
                        print(f"No Stiffness attribute found for {prim.GetPath()}.")

                    damping_attr = drive_api.GetDampingAttr()
                    if damping_attr:
                        damping_attr.Set(100000.0)
                        print(f"Damping for {prim.GetPath()} set to 100000.0.")
                    else:
                        print(f"No Damping attribute found for {prim.GetPath()}.")

                    force_attr = drive_api.GetMaxForceAttr()
                    if force_attr:
                        force_attr.Set(1000.0)
                        print(f"Max Force for {prim.GetPath()} set to 1000.0.")
                    else:
                        print(f"No Force attribute found for {prim.GetPath()}.")
                else:
                    print(f"No Drive found at {prim.GetPath()}.")

    stage.GetRootLayer().Save()

    # Reinitialize the simulation
    app = omni.kit.app.get_app_interface()
    run_gui = args_cli.run_gui
    if run_gui:
        # Run simulation
        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():
                # perform step
                app.update()
    else:
        app.update()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
