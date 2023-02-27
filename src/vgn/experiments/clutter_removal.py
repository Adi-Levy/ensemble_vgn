import collections
from datetime import datetime
import uuid
import os

import numpy as np
import pandas as pd
import tqdm

from vgn import io, vis
from vgn.grasp import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform

MAX_CONSECUTIVE_FAILURES = 2


State = collections.namedtuple("State", ["tsdf", "pc"])


def run(
    grasp_plan_fn,
    logdir,
    description,
    scene,
    object_set,
    num_objects=5,
    n=6,
    N=None,
    num_rounds=40,
    seed=1,
    sim_gui=False,
    rviz=False,
    ensemble_type=None
):
    """Run several rounds of simulated clutter removal experiments.

    Each round, m objects are randomly placed in a tray. Then, the grasping pipeline is
    run until (a) no objects remain, (b) the planner failed to find a grasp hypothesis,
    or (c) maximum number of consecutive failed grasp attempts.
    """
    sim = ClutterRemovalSim(scene, object_set, gui=sim_gui, seed=seed, train=False)
    logger = Logger(logdir, description)

    # create a timestamped directory to store the grasps
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = f"sim_res/grasps_{timestamp}_{uuid.uuid4()}"
    os.makedirs(dir_name)
    
    for round_id in tqdm.tqdm(range(num_rounds)):
        first_run = True
        # round_id == 0
        sim.reset(num_objects, first_run=first_run)

        round_id = logger.last_round_id() + 1
        logger.log_round(round_id, sim.num_objects)

        consecutive_failures = 1
        last_label = None

        attempt = 0
        while sim.num_objects > 0 and consecutive_failures < MAX_CONSECUTIVE_FAILURES:
            timings = {}

            # scan the scene
            tsdf, pc, timings["integration"] = sim.acquire_tsdf(n=n, N=N)

            if pc.is_empty():
                break  # empty point cloud, abort this round TODO this should not happen

            # visualize scene
            if rviz:
                vis.clear()
                vis.draw_workspace(sim.size)
                vis.draw_tsdf(tsdf.get_grid().squeeze(), tsdf.voxel_size)
                vis.draw_points(np.asarray(pc.points))

            # plan grasps
            state = State(tsdf, pc)
            grasps, scores, timings["planning"] = grasp_plan_fn(state)
            grsps = np.array([np.concatenate((np.concatenate((grasp.pose.translation, grasp.pose.rotation.as_euler('xyz', degrees=True)), axis=-1), np.array([grasp.width])), axis=-1) for grasp in grasps])
            data = []
            for i in range(len(grasps)):
                data.append(np.concatenate([grsps[i], np.array([scores[i]], dtype=np.float64)]))
            data = np.array(data, dtype=np.float64)
            # save the grasps to the created directory
            if ensemble_type != None:
                np.save(f"{dir_name}/grasps_{round_id}_{attempt}_enseble_by_{ensemble_type}.npy", data)
            else:
                np.save(f"{dir_name}/grasps_{round_id}_{attempt}.npy", data)

            if len(grasps) == 0:
                break  # no detections found, abort this round

            if rviz:
                vis.draw_grasps(grasps, scores, sim.gripper.finger_depth)

            # execute grasp
            grasp, score = grasps[0], scores[0]
            if rviz:
                vis.draw_grasp(grasp, score, sim.gripper.finger_depth)
            # for grasp, score in zip(grasps, scores):
            #     if rviz:
            #         sim.draw_poc(grasp.pose.translation)
            # grasp, score = grasps[0], scores[0]
            label, _ = sim.execute_grasp(grasp, allow_contact=True)

            # log the grasp
            logger.log_grasp(round_id, state, timings, grasp, score, label)

            if last_label == Label.FAILURE and label == Label.FAILURE:
                consecutive_failures += 1
            else:
                consecutive_failures = 1
            last_label = label

            # if rviz:
                # sim.erase_pocs()

            attempt += 1


class Logger(object):
    def __init__(self, root, description):
        time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        description = "{}_{}".format(time_stamp, description).strip("_")

        self.logdir = root / description
        self.scenes_dir = self.logdir / "scenes"
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

        self.rounds_csv_path = self.logdir / "rounds.csv"
        self.grasps_csv_path = self.logdir / "grasps.csv"
        self._create_csv_files_if_needed()

    def _create_csv_files_if_needed(self):
        if not self.rounds_csv_path.exists():
            io.create_csv(self.rounds_csv_path, ["round_id", "object_count"])

        if not self.grasps_csv_path.exists():
            columns = [
                "round_id",
                "scene_id",
                "qx",
                "qy",
                "qz",
                "qw",
                "x",
                "y",
                "z",
                "width",
                "score",
                "label",
                "integration_time",
                "planning_time",
            ]
            io.create_csv(self.grasps_csv_path, columns)

    def last_round_id(self):
        df = pd.read_csv(self.rounds_csv_path)
        return -1 if df.empty else df["round_id"].max()

    def log_round(self, round_id, object_count):
        io.append_csv(self.rounds_csv_path, round_id, object_count)

    def log_grasp(self, round_id, state, timings, grasp, score, label):
        # log scene
        tsdf, points = state.tsdf, np.asarray(state.pc.points)
        scene_id = uuid.uuid4().hex
        scene_path = self.scenes_dir / (scene_id + ".npz")
        np.savez_compressed(scene_path, grid=tsdf.get_grid(), points=points)

        # log grasp
        qx, qy, qz, qw = grasp.pose.rotation.as_quat()
        x, y, z = grasp.pose.translation
        width = grasp.width
        label = int(label)
        io.append_csv(
            self.grasps_csv_path,
            round_id,
            scene_id,
            qx,
            qy,
            qz,
            qw,
            x,
            y,
            z,
            width,
            score,
            label,
            timings["integration"],
            timings["planning"],
        )


class Data(object):
    """Object for loading and analyzing experimental data."""

    def __init__(self, logdir):
        self.logdir = logdir
        self.rounds = pd.read_csv(logdir / "rounds.csv")
        self.grasps = pd.read_csv(logdir / "grasps.csv")

    def num_rounds(self):
        return len(self.rounds.index)

    def num_grasps(self):
        return len(self.grasps.index)

    def success_rate(self):
        return self.grasps["label"].mean() * 100

    def percent_cleared(self):
        df = (
            self.grasps[["round_id", "label"]]
            .groupby("round_id")
            .sum()
            .rename(columns={"label": "cleared_count"})
            .merge(self.rounds, on="round_id")
        )
        return df["cleared_count"].sum() / df["object_count"].sum() * 100

    def avg_planning_time(self):
        return self.grasps["planning_time"].mean()

    def read_grasp(self, i):
        scene_id, grasp, label = io.read_grasp(self.grasps, i)
        score = self.grasps.loc[i, "score"]
        scene_data = np.load(self.logdir / "scenes" / (scene_id + ".npz"))

        return scene_data["points"], grasp, score, label
