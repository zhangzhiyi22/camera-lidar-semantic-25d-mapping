"""Collect a synchronized CARLA camera/LiDAR dataset for 2.5D semantic mapping.

The output layout is directly consumable by:
  camera-lidar-semantic-25d-mapping/scripts/fuse_camera_lidar_semantic_25d.py
"""

import argparse
import json
import math
import queue
import random
from pathlib import Path

import carla
import numpy as np


CARLA_TO_CAMERA = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
DETECTION_CLASSES = ("car", "truck", "bus", "motorcycle", "bicycle", "pedestrian")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect synchronous CARLA RGB/LiDAR data for semantic 2.5D mapping."
    )
    parser.add_argument("--output-dir", default="camera_lidar_dataset")
    parser.add_argument("--frames", type=int, default=300, help="Frames to save at 10 Hz.")
    parser.add_argument("--warmup-ticks", type=int, default=20)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", dest="map_name", default=None, help="Optional CARLA map, e.g. Town05.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--rig-height", type=float, default=30.0)
    parser.add_argument("--pitch", type=float, default=-75.0)
    parser.add_argument("--lidar-range", type=float, default=80.0)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--traffic-vehicles", type=int, default=50)
    parser.add_argument("--walkers", type=int, default=30)
    return parser.parse_args()


def transform_matrix(transform):
    return np.asarray(transform.get_matrix(), dtype=np.float64)


def camera_intrinsics(width, height, fov_deg):
    focal = width / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    return [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]


def write_ascii_ply(measurement, output):
    points = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)
    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "property float intensity",
            "end_header",
        ]
    )
    np.savetxt(output, points, fmt="%.6f", header=header, comments="")


def get_frame_data(sensor_queues, expected_frame, timeout=5.0):
    data = {}
    for name, sensor_queue in sensor_queues.items():
        while True:
            measurement = sensor_queue.get(timeout=timeout)
            if measurement.frame == expected_frame:
                data[name] = measurement
                break
            if measurement.frame > expected_frame:
                raise RuntimeError(
                    f"{name} skipped frame {expected_frame}; received {measurement.frame}. "
                    "Check that all sensors use the same sensor_tick."
                )
    return data


def make_sensor(world, blueprint, transform, parent, sensor_queues, name):
    sensor = world.spawn_actor(blueprint, transform, attach_to=parent)
    sensor.listen(sensor_queues[name].put)
    return sensor


def actor_category(actor):
    type_id = actor.type_id.lower()
    if type_id.startswith("walker.pedestrian"):
        return "pedestrian"
    if not type_id.startswith("vehicle."):
        return None
    if "bus" in type_id:
        return "bus"
    if any(token in type_id for token in ("truck", "van", "cargo")):
        return "truck"
    if "motorcycle" in type_id:
        return "motorcycle"
    if "bicycle" in type_id or "bike" in type_id:
        return "bicycle"
    return "car"


def collect_3d_annotations(world, lidar_transform, lidar_range, ego_actor_id):
    """Export dynamic CARLA actors as LiDAR-frame 3D boxes for BEV training."""
    world_from_lidar = transform_matrix(lidar_transform)
    lidar_from_world = np.linalg.inv(world_from_lidar)
    annotations = []
    for actor in world.get_actors():
        if actor.id == ego_actor_id or not actor.is_alive:
            continue
        category = actor_category(actor)
        if category is None:
            continue
        bbox = actor.bounding_box
        center_world = (transform_matrix(actor.get_transform()) @ np.array([bbox.location.x, bbox.location.y, bbox.location.z, 1.0]))[:3]
        center_lidar = (lidar_from_world @ np.r_[center_world, 1.0])[:3]
        size = np.array([bbox.extent.x * 2.0, bbox.extent.y * 2.0, bbox.extent.z * 2.0])
        if np.linalg.norm(center_lidar[:2]) > lidar_range + np.linalg.norm(size[:2]):
            continue
        world_yaw = math.radians(actor.get_transform().rotation.yaw + bbox.rotation.yaw)
        lidar_yaw = math.atan2(math.sin(world_yaw - math.radians(lidar_transform.rotation.yaw)), math.cos(world_yaw - math.radians(lidar_transform.rotation.yaw)))
        velocity = actor.get_velocity()
        annotations.append(
            {
                "actor_id": actor.id,
                "category": category,
                "bbox_lidar": [*center_lidar.tolist(), *size.tolist(), lidar_yaw],
                "bbox_world": [*center_world.tolist(), *size.tolist(), world_yaw],
                "velocity_world_mps": [velocity.x, velocity.y, velocity.z],
            }
        )
    return annotations


def main():
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive.")

    random.seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    paths = {
        name: output_dir / name
        for name in ("camera", "lidar", "poses", "semantic", "depth", "labels")
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    if args.map_name:
        world = client.load_world(args.map_name)
    else:
        world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    original_settings = world.get_settings()
    actor_list = []
    traffic_vehicles = []
    walker_controllers = []

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)
    traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(args.seed)

    try:
        vehicle_bp = random.choice(blueprint_library.filter("vehicle.*"))
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("The selected map has no vehicle spawn points.")
        vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        if vehicle is None:
            raise RuntimeError("Could not spawn the collection vehicle; run the script again or choose another map.")
        actor_list.append(vehicle)
        vehicle.set_autopilot(True, traffic_manager.get_port())

        # In synchronous mode, Traffic Manager must also be synchronous or the
        # autopilot can remain at the spawn point for every captured frame.
        background_spawns = list(spawn_points)
        random.shuffle(background_spawns)
        for traffic_spawn in background_spawns:
            if len(actor_list) - 1 >= args.traffic_vehicles:
                break
            if traffic_spawn.location.distance(vehicle.get_location()) < 8.0:
                continue
            traffic_bp = random.choice(blueprint_library.filter("vehicle.*"))
            traffic_vehicle = world.try_spawn_actor(traffic_bp, traffic_spawn)
            if traffic_vehicle is not None:
                traffic_vehicle.set_autopilot(True, traffic_manager.get_port())
                actor_list.append(traffic_vehicle)
                traffic_vehicles.append(traffic_vehicle)

        walker_bp_choices = blueprint_library.filter("walker.pedestrian.*")
        walker_controller_bp = blueprint_library.find("controller.ai.walker")
        for _ in range(args.walkers):
            location = world.get_random_location_from_navigation()
            if location is None:
                continue
            walker = world.try_spawn_actor(random.choice(walker_bp_choices), carla.Transform(location))
            if walker is None:
                continue
            controller = world.try_spawn_actor(walker_controller_bp, carla.Transform(), attach_to=walker)
            if controller is None:
                walker.destroy()
                continue
            actor_list.extend((walker, controller))
            walker_controllers.append(controller)

        world.tick()
        for controller in walker_controllers:
            controller.start()
            destination = world.get_random_location_from_navigation()
            if destination is not None:
                controller.go_to_location(destination)
            controller.set_max_speed(random.uniform(1.0, 1.8))

        rig_transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=args.rig_height),
            carla.Rotation(pitch=args.pitch),
        )
        sensor_queues = {name: queue.Queue() for name in ("rgb", "lidar", "semantic", "depth")}

        rgb_bp = blueprint_library.find("sensor.camera.rgb")
        sem_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
        depth_bp = blueprint_library.find("sensor.camera.depth")
        for camera_bp in (rgb_bp, sem_bp, depth_bp):
            camera_bp.set_attribute("image_size_x", str(args.width))
            camera_bp.set_attribute("image_size_y", str(args.height))
            camera_bp.set_attribute("fov", str(args.fov))
            camera_bp.set_attribute("sensor_tick", "0.1")

        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", "64")
        lidar_bp.set_attribute("range", str(args.lidar_range))
        lidar_bp.set_attribute("rotation_frequency", "10")
        lidar_bp.set_attribute("points_per_second", "250000")
        lidar_bp.set_attribute("upper_fov", "10")
        lidar_bp.set_attribute("lower_fov", "-30")
        lidar_bp.set_attribute("sensor_tick", "0.1")

        rgb = make_sensor(world, rgb_bp, rig_transform, vehicle, sensor_queues, "rgb")
        lidar = make_sensor(world, lidar_bp, rig_transform, vehicle, sensor_queues, "lidar")
        semantic = make_sensor(world, sem_bp, rig_transform, vehicle, sensor_queues, "semantic")
        depth = make_sensor(world, depth_bp, rig_transform, vehicle, sensor_queues, "depth")
        actor_list.extend((rgb, lidar, semantic, depth))

        for _ in range(args.warmup_ticks):
            world.tick()
        lidar_to_camera = CARLA_TO_CAMERA @ np.linalg.inv(transform_matrix(rgb.get_transform())) @ transform_matrix(lidar.get_transform())
        calibration = {
            "camera_intrinsics": camera_intrinsics(args.width, args.height, args.fov),
            "lidar_to_camera": lidar_to_camera.tolist(),
            "metadata": {
                "carla_version": client.get_server_version(),
                "fixed_delta_seconds": 0.1,
                "rig_height_m": args.rig_height,
                "rig_pitch_deg": args.pitch,
                "lidar_range_m": args.lidar_range,
                "background_traffic_vehicles": len(traffic_vehicles),
                "background_walkers": len(walker_controllers),
                "coordinate_note": "poses use CARLA world_from_lidar; lidar_to_camera uses standard pinhole camera axes (right, down, forward).",
            },
        }
        (output_dir / "calib.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        label_schema = {
            "classes": DETECTION_CLASSES,
            "bbox_lidar_layout": ["x", "y", "z", "length", "width", "height", "yaw_radians"],
            "bbox_world_layout": ["x", "y", "z", "length", "width", "height", "yaw_radians"],
            "note": "Boxes are CARLA simulator ground truth for dynamic actors within LiDAR range; use them only for offline training and evaluation.",
        }
        (output_dir / "label_schema.json").write_text(json.dumps(label_schema, indent=2), encoding="utf-8")

        print(f"Collecting {args.frames} synchronized frames into {output_dir} with {len(traffic_vehicles)} traffic vehicles and {len(walker_controllers)} walkers")
        for index in range(args.frames):
            frame = world.tick()
            measurements = get_frame_data(sensor_queues, frame)
            stem = f"{index:06d}"
            measurements["rgb"].save_to_disk(str(paths["camera"] / f"{stem}.png"))
            measurements["semantic"].save_to_disk(str(paths["semantic"] / f"{stem}.png"))
            measurements["depth"].save_to_disk(str(paths["depth"] / f"{stem}.png"))
            write_ascii_ply(measurements["lidar"], paths["lidar"] / f"{stem}.ply")
            np.savetxt(paths["poses"] / f"{stem}.txt", transform_matrix(lidar.get_transform()), fmt="%.10f")
            annotations = collect_3d_annotations(world, lidar.get_transform(), args.lidar_range, vehicle.id)
            label_payload = {"frame_index": index, "simulator_frame": frame, "timestamp_s": measurements["rgb"].timestamp, "annotations": annotations}
            (paths["labels"] / f"{stem}.json").write_text(json.dumps(label_payload, indent=2), encoding="utf-8")
            print(f"[{index + 1:04d}/{args.frames}] frame {frame} saved as {stem}; labels={len(annotations)}")

    finally:
        for actor in reversed(actor_list):
            if actor.is_alive:
                actor.destroy()
        traffic_manager.set_synchronous_mode(False)
        world.apply_settings(original_settings)
        print("CARLA settings restored and spawned actors removed.")


if __name__ == "__main__":
    main()
