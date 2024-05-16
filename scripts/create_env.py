from __future__ import annotations

import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test adding sensors on a robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--num_cams", type=int, default=1, help="Number of cams per env (2 Max)")
parser.add_argument("--save", action="store_true", default=False, help="Save the obtained data to disk.")
# parser.add_argument("--livestream", type=int, default="1", help="stream remotely")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.num_cams = min(2, args_cli.num_cams)
args_cli.num_cams = max(0, args_cli.num_cams)
args_cli.num_envs = max(1, args_cli.num_envs)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import math
from PIL import Image
import torch
import traceback

import carb
import os
import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.orbit.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.orbit.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.orbit.actuators import ImplicitActuatorCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.timer import Timer
import omni.replicator.core as rep
from omni.isaac.orbit.utils import convert_dict_to_backend
from tqdm import tqdm

current_script_path = os.path.abspath(__file__)
# Absolute path of the project root (assuming it's three levels up from the current script)
project_root = os.path.join(current_script_path, "../..")

MAZE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAAC_ORBIT_NUCLEUS_DIR}/Robots/Classic/Cartpole/cartpole.usd",
        # Path to the USD file relative to the project root
        usd_path=os.path.join(project_root, "usds/Maze_Simple.usd"),
        # usd_path=f"../../../../usds/Maze_Simple.usd",
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
    actuators={
        "outer_actuator": ImplicitActuatorCfg(
            joint_names_expr=["OuterDOF_RevoluteJoint"],
            effort_limit=0.01,
            velocity_limit=1.0 / math.pi,
            stiffness=0.0,
            damping=10.0,
        ),
        "inner_actuator": ImplicitActuatorCfg(
            joint_names_expr=["InnerDOF_RevoluteJoint"],
            effort_limit=0.01,
            velocity_limit=1.0 / math.pi,
            stiffness=0.0,
            damping=10.0,
        ),
    },
)

@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

 # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # cartpole
    robot: ArticulationCfg = MAZE_CFG.replace(prim_path="{ENV_REGEX_NS}/Labyrinth")

    # Sphere with collision enabled but not actuated
    sphere = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.005,  # Define the radius of the sphere
            mass_props=sim_utils.MassPropertiesCfg(density=7850),  # Density of steel in kg/m^3)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9), metallic=0.8),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.11)),
    )
    
    # sensors
    camera_1 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/top_cam",
        update_period=0.1,
        height=8,
        width=8,
        data_types=["rgb"],#, "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.5), rot=(0,1,0,0), convention="ros"),
    )
    
    # sphere_object = RigidObject(cfg=sphere_cfg)

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
):
    """Run the simulator."""

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    
    def reset():
        # reset the scene entities
        # root state
        # we offset the root state by the origin since the states are written in simulation world frame
        # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
        root_state = scene["robot"].data.default_root_state.clone()
        root_state[:, :3] += scene.env_origins
        scene["robot"].write_root_state_to_sim(root_state)
        # set joint positions with some noise
        joint_pos, joint_vel = (
            scene["robot"].data.default_joint_pos.clone(),
            scene["robot"].data.default_joint_vel.clone(),
        )
        joint_pos += torch.rand_like(joint_pos) * 0.1
        scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
        # clear internal buffers
        scene.reset()
        print("[INFO]: Resetting robot state...")

    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=3)
    
    episode_steps = 500

    while simulation_app.is_running():
        reset()
        
        with Timer(f"Time taken for {episode_steps} steps with {args_cli.num_envs} envs"):
            with tqdm(range(episode_steps*args_cli.num_envs)) as pbar:
                for count in range(episode_steps):
                    # Apply default actions to the robot
                    # -- generate actions/commands
                    targets = scene["robot"].data.default_joint_pos
                    # -- apply action to the robot
                    scene["robot"].set_joint_position_target(targets)
                    # -- write data to sim
                    scene.write_data_to_sim()
                    # perform step
                    sim.step()
                    # update sim-time
                    sim_time += sim_dt
                    count += 1
                    # update buffers
                    scene.update(sim_dt)
                    pbar.update(args_cli.num_envs)
                    
                    # Extract camera data
                    if args_cli.save:
                        for i in range(args_cli.num_envs):
                            for j in range(args_cli.num_cams):
                                single_cam_data = convert_dict_to_backend(scene[f"camera_{j+1}"].data.output, backend="numpy")
                                #single_cam_info = scene[f"camera_{j+1}"].data.info

                                # Pack data back into replicator format to save them using its writer
                                rep_output = dict()
                                for key, data in zip(single_cam_data.keys(), single_cam_data.values()):#, single_cam_info):
                                    # if info is not None:
                                    #     rep_output[key] = {"data": data, "info": info}
                                    # else:
                                    rep_output[key] = data[i]
                                # Save images
                                # Note: We need to provide On-time data for Replicator to save the images.
                                rep_output["trigger_outputs"] = {"on_time":f"{count}_{i}_{j}"}#{"on_time": scene["camera_1"].frame}
                                rep_writer.write(rep_output)
                    
                    if args_cli.num_cams > 0:
                        cam1_rgb = scene["camera_1"].data.output["rgb"]
                        squeezed_img = cam1_rgb.squeeze(0).cpu().numpy().astype('uint8')
                        image = Image.fromarray(squeezed_img)
                        # image.save('test_cam'+str(count)+'.png')
                    if args_cli.num_cams > 1:
                        cam2_rgb = scene["camera_2"].data.output["rgb"]
                    
                
def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, substeps=1)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = SensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
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