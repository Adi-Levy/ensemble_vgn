import time
import os

import numpy as np
from scipy import ndimage
import torch

from vgn import vis
from vgn.grasp import *
from vgn.utils.transform import Transform, Rotation
from vgn.networks import load_network, EnsembleConvNet


class VGN(object):
    def __init__(self, model_path, rviz=False, ensemble_type=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(model_path, self.device)
        self.ensemble = False
        if isinstance(self.net, EnsembleConvNet):
            self.ensemble = True
            self.ensemble_type = ensemble_type
        self.rviz = rviz
        self.id = 0

    def __call__(self, state):
        tsdf_vol = state.tsdf.get_grid()
        voxel_size = state.tsdf.voxel_size

        tic = time.time()
        qual_vol, rot_vol, width_vol = predict(tsdf_vol, self.net, self.device)
        if self.ensemble:
            # This is the actual project code
            # raise NotImplementedError('Ensemble not implemented yet.')
            for i in range(0, len(qual_vol)):
                qual_vol[i], rot_vol, width_vol = process(tsdf_vol, qual_vol[i], rot_vol, width_vol)
                # data = np.concatenate([np.expand_dims(qual_vol[i].copy(),0), rot_vol, np.expand_dims(width_vol.copy(),0)], axis=0)
                # np.save(os.path.join(os.getcwd(),f"data/grasps/grasps_{self.id%5}_{self.id//5}"), data)
                # self.id += 1
            
            qual = np.transpose(qual_vol, (1,2,3,0))
            # max ensemble between qualities
            if self.ensemble_type == "max":
                qual = np.max(qual,axis=3)
            # UCB ensemble between qualities
            elif self.ensemble_type == "ucb":
                qual = np.mean(qual, axis=3) + np.var(qual, axis=3)
            # mean-var ensemble between qualities
            elif self.ensemble_type == "mean-var":
                qual = np.mean(qual, axis=3) - np.var(qual, axis=3)
            else:
                raise NotImplementedError('when using ensemble you must select one 3 methods: max, ucb, maen-var')

            data = np.concatenate([np.expand_dims(qual.copy(),0), rot_vol, np.expand_dims(width_vol.copy(),0)], axis=0)
            np.save(os.path.join(os.getcwd(),f"data/grasps/grasps_{self.id}"), data)
            self.id += 1
            
            # qual = np.transpose(qual, (3,0,1,2))
            grasps, scores = select(qual.copy(), rot_vol, width_vol)
            # graspss.append(grasps)
            # scoress.append(scores)
            toc = time.time() - tic

            grasps, scores = np.asarray(grasps), np.asarray(scores)

            if len(grasps) > 0:
                p = np.random.permutation(len(grasps))
                grasps = [from_voxel_coordinates(g, voxel_size) for g in grasps[p]]
                scores = scores[p]
            # if len(grasps) > 0:
            #     p = np.flip(np.argsort(scores))
            #     scores = np.flip(np.sort(scores))
            #     grasps = [from_voxel_coordinates(grasps[i], voxel_size) for i in p]
            
            if self.rviz:
                vis.draw_quality(qual_vol, state.tsdf.voxel_size, threshold=0.01)

            return grasps, scores, toc
        else:
            qual_vol, rot_vol, width_vol = process(tsdf_vol, qual_vol, rot_vol, width_vol)
            data = np.concatenate([np.expand_dims(qual_vol.copy(),0), rot_vol, np.expand_dims(width_vol.copy(),0)], axis=0)
            np.save(os.path.join(os.getcwd(),f"data/grasps/grasps_{self.id}"), data)
            self.id += 1
            grasps, scores = select(qual_vol.copy(), rot_vol, width_vol)
            toc = time.time() - tic

            grasps, scores = np.asarray(grasps), np.asarray(scores)

            if len(grasps) > 0:
                p = np.random.permutation(len(grasps))
                grasps = [from_voxel_coordinates(g, voxel_size) for g in grasps[p]]
                scores = scores[p]

            if self.rviz:
                vis.draw_quality(qual_vol, state.tsdf.voxel_size, threshold=0.01)

            return grasps, scores, toc


def predict(tsdf_vol, net, device, validate=False):
    if not validate:
        assert tsdf_vol.shape == (1, 40, 40, 40)

    # move input to the GPU if needed
    if isinstance(tsdf_vol, torch.Tensor):
        tsdf_vol = tsdf_vol.to(device)
    else:
        tsdf_vol = torch.from_numpy(tsdf_vol).unsqueeze(0).to(device)

    # forward pass
    with torch.no_grad():
        qual_vol, rot_vol, width_vol = net(tsdf_vol)

    # move output back to the CPU
    qual_vol = qual_vol.cpu().squeeze().numpy()
    rot_vol = rot_vol.cpu().squeeze().numpy()
    width_vol = width_vol.cpu().squeeze().numpy()
    return qual_vol, rot_vol, width_vol


def process(
    tsdf_vol,
    qual_vol,
    rot_vol,
    width_vol,
    gaussian_filter_sigma=1.0,
    min_width=1.33,
    max_width=9.33,
):
    tsdf_vol = tsdf_vol.squeeze()

    # smooth quality volume with a Gaussian
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )

    # mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > 0.5
    inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < 0.5)
    valid_voxels = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    qual_vol[valid_voxels == False] = 0.0

    # reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    return qual_vol, rot_vol, width_vol


def select(qual_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4):
    # threshold on grasp quality
    qual_vol[qual_vol < threshold] = 0.0

    # non maximum suppression
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask = np.where(qual_vol, 1.0, 0.0)

    # construct grasps
    grasps, scores = [], []
    for index in np.argwhere(mask):
        grasp, score = select_index(qual_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)

    return grasps, scores


def select_index(qual_vol, rot_vol, width_vol, index):
    i, j, k = index
    score = qual_vol[i, j, k]
    ori = Rotation.from_quat(rot_vol[:, i, j, k])
    pos = np.array([i, j, k], dtype=np.float64)
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score
