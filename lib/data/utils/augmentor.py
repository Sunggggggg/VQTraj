from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from configs import constants as _C

import torch
import numpy as np
from torch.nn import functional as F

from ...utils import transforms

__all__ = ['VideoAugmentor', 'SMPLAugmentor', 'SequenceAugmentor', 'CameraAugmentor']


num_joints = _C.KEYPOINTS.NUM_JOINTS
class VideoAugmentor():
    def __init__(self, cfg, train=True):
        self.train = train
        self.l = cfg.DATASET.SEQLEN + 1
        self.aug_dict = torch.load(_C.KEYPOINTS.COCO_AUG_DICT)        

    def get_jitter(self, ):
        """Guassian jitter modeling."""
        jittering_noise = torch.normal(
            mean=torch.zeros((self.l, num_joints, 3)),
            std=self.aug_dict['jittering'].reshape(1, num_joints, 1).expand(self.l, -1, 3)
        ) * _C.KEYPOINTS.S_JITTERING
        return jittering_noise

    def get_lfhp(self, ):
        """Low-frequency high-peak noise modeling."""
        def get_peak_noise_mask():
            peak_noise_mask = torch.rand(self.l, num_joints).float() * self.aug_dict['pmask'].squeeze(0)
            peak_noise_mask = peak_noise_mask < _C.KEYPOINTS.S_PEAK_MASK
            return peak_noise_mask

        peak_noise_mask = get_peak_noise_mask()
        peak_noise = peak_noise_mask.float().unsqueeze(-1).repeat(1, 1, 3)
        peak_noise = peak_noise * torch.randn(3) * self.aug_dict['peak'].reshape(1, -1, 1) * _C.KEYPOINTS.S_PEAK
        return peak_noise

    def get_bias(self, ):
        """Bias noise modeling."""
        bias_noise = torch.normal(
            mean=torch.zeros((num_joints, 3)), std=self.aug_dict['bias'].reshape(num_joints, 1)
        ).unsqueeze(0) * _C.KEYPOINTS.S_BIAS
        return bias_noise
    
    def get_mask(self, scale=None):
        """Mask modeling."""

        if scale is None:
            scale = _C.KEYPOINTS.S_MASK
        # Per-frame and joint
        mask = torch.rand(self.l, num_joints) < scale
        visible = (~mask).clone()
        for child in range(num_joints):
            parent = _C.KEYPOINTS.TREE[child]
            if parent == -1: continue
            if isinstance(parent, list):
                visible[:, child] *= (visible[:, parent[0]] * visible[:, parent[1]])
            else:
                visible[:, child] *= visible[:, parent]
        mask = (~visible).clone()

        return mask

    def __call__(self, keypoints):
        keypoints += self.get_bias() + self.get_jitter() + self.get_lfhp()
        return keypoints


class SMPLAugmentor():
    noise_scale = 1e-2

    def __init__(self, cfg, augment=True):
        self.n_frames = cfg.DATASET.SEQLEN
        self.augment = augment

    def __call__(self, target):
        if not self.augment:
            return target

        n_frames = target['body_pose'].shape[0]

        # Global rotation
        rmat = self.get_global_augmentation()
        target['w_root_orient'][:, 0] = rmat @ target['w_root_orient'][:, 0]
        target['w_transl'] = (rmat.squeeze() @ target['w_transl'].T).T

        # Shape
        shape_noise = self.get_shape_augmentation(n_frames)
        target['betas'] = target['betas'] + shape_noise
        
        return target

    def get_global_augmentation(self, ):
        """Global coordinate augmentation. Random rotation around y-axis"""
        
        angle_y = torch.rand(1) * 2 * np.pi * float(self.augment)
        aa = torch.tensor([0.0, angle_y, 0.0]).float().unsqueeze(0)
        rmat = transforms.axis_angle_to_matrix(aa)

        return rmat

    def get_shape_augmentation(self, n_frames):
        """Shape noise modeling."""
        
        shape_noise = torch.normal(
            mean=torch.zeros((1, 10)),
            std=torch.ones((1, 10)) * 0.1 * float(self.augment)).expand(n_frames, 10)

        return shape_noise

    def get_initial_pose_augmentation(self, ):
        """Initial frame pose noise modeling. Random rotation around all joints."""
        
        euler = torch.normal(
            mean=torch.zeros((24, 3)),
            std=torch.ones((24, 3))
        ) * self.noise_scale #* float(self.augment)
        rmat = transforms.axis_angle_to_matrix(euler)

        return rmat.unsqueeze(0)


class SequenceAugmentor:
    """Augment the play speed of the motion sequence"""
    l_factor = 1.5
    def __init__(self, l_default):
        self.l_default = l_default

    def __call__(self, target):
        l = torch.randint(low=int(self.l_default / self.l_factor), high=int(self.l_default * self.l_factor), size=(1, ))

        pose = transforms.matrix_to_rotation_6d(target['pose'])
        resampled_pose = F.interpolate(
            pose[:l].permute(1, 2, 0), self.l_default, mode='linear', align_corners=True
        ).permute(2, 0, 1)
        resampled_pose = transforms.rotation_6d_to_matrix(resampled_pose)

        transl = target['transl'].unsqueeze(1)
        resampled_transl = F.interpolate(
            transl[:l].permute(1, 2, 0), self.l_default, mode='linear', align_corners=True
        ).squeeze(0).T
        
        target['pose'] = resampled_pose
        target['transl'] = resampled_transl
        target['betas'] = target['betas'][:self.l_default]
        
        return target
    
    
class CameraAugmentor:
    rx_factor = np.pi/8
    ry_factor = np.pi/4
    rz_factor = np.pi/8
    
    pitch_std = np.pi/8
    pitch_mean = np.pi/36
    roll_std = np.pi/24
    t_factor = 1
    
    tz_scale = 10
    tz_min = 2
    
    motion_prob = 0.75
    interp_noise = 0.2
    
    def __init__(self, l, w, h, f):
        self.l = l
        self.w = w
        self.h = h
        self.f = f
        self.fov_tol = 1.2 * (0.5 ** 0.5)
        
    def __call__(self, target):
        
        R, T = self.create_camera(target)           # c2w, global translation
        
        if np.random.rand() < self.motion_prob:
            R = self.create_rotation_move(R)
            T = self.create_translation_move(T)
        
        target['R'] = R
        target['T'] = T
        return target
        
    def create_camera(self, target):
        """Create the initial frame camera pose"""
        yaw = np.random.rand() * 2 * np.pi
        pitch = np.random.normal(scale=self.pitch_std) + self.pitch_mean
        roll = np.random.normal(scale=self.roll_std)
        
        yaw_rm = transforms.axis_angle_to_matrix(torch.tensor([[0, yaw, 0]]).float())
        pitch_rm = transforms.axis_angle_to_matrix(torch.tensor([[pitch, 0, 0]]).float())
        roll_rm = transforms.axis_angle_to_matrix(torch.tensor([[0, 0, roll]]).float())
        R = (roll_rm @ pitch_rm @ yaw_rm)
        
        # Place people in the scene
        tz = np.random.rand() * self.tz_scale + self.tz_min
        max_d = self.w * tz / self.f / 2
        tx = np.random.normal(scale=0.25) * max_d
        ty = np.random.normal(scale=0.25) * max_d
        dist = torch.tensor([tx, ty, tz]).float() 
        T = dist - torch.matmul(R, target['w_transl'][0]) # c2w / target['transl'] : Global body position
        
        return R.repeat(self.l, 1, 1), T.repeat(self.l, 1)
    
    def create_rotation_move(self, R):
        """Create rotational move for the camera"""
        
        # Create final camera pose
        rx = np.random.normal(scale=self.rx_factor)
        ry = np.random.normal(scale=self.ry_factor)
        rz = np.random.normal(scale=self.rz_factor)
        Rf = R[0] @ transforms.axis_angle_to_matrix(torch.tensor([rx, ry, rz]).float())
        
        # Inbetweening two poses
        Rs = torch.stack((R[0], Rf))
        rs = transforms.matrix_to_rotation_6d(Rs).numpy() 
        rs_move = self.noisy_interpolation(rs)
        R_move = transforms.rotation_6d_to_matrix(torch.from_numpy(rs_move).float())
        return R_move
    
    def create_translation_move(self, T):
        """Create translational move for the camera"""
        
        # Create final camera position
        tx = np.random.normal(scale=self.t_factor)
        ty = np.random.normal(scale=self.t_factor)
        tz = np.random.normal(scale=self.t_factor)
        Ts = np.array([[0, 0, 0], [tx, ty, tz]])
        
        T_move = self.noisy_interpolation(Ts)
        T_move = torch.from_numpy(T_move).float()
        return T_move + T
        
    def noisy_interpolation(self, data):
        """Non-linear interpolation with noise"""
        
        dim = data.shape[-1]
        output = np.zeros((self.l, dim))
        
        linspace = np.stack([np.linspace(0, 1, self.l) for _ in range(dim)])
        noise = (linspace[0, 1] - linspace[0, 0]) * self.interp_noise
        space_noise = np.stack([np.random.uniform(-noise, noise, self.l - 2) for _ in range(dim)])
        
        linspace[:, 1:-1] = linspace[:, 1:-1] + space_noise
        for i in range(dim):
            output[:, i] = np.interp(linspace[i], np.array([0., 1.,]), data[:, i])
        return output