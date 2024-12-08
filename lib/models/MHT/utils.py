import torch

def World2Camera(w_kp3d_coco, R, c_transl):
    kp3d = torch.matmul(R, w_kp3d_coco.transpose(-1, -2)).transpose(-1, -2) + c_transl.unsqueeze(-2)
    return kp3d

def get_virtual_camera(R, T, w_transl, w_orinet):
    fov_tol = 1.2 * (0.5 ** 0.5)

    c_transl_list, c_orinet_list = [], []
    for _R, _T, _transl, _orient in zip(R, T, w_transl, w_orinet):    
        # Recompute the translation ()
        transl_cam = torch.matmul(_R, _transl.unsqueeze(-1)).squeeze(-1)
        transl_cam = transl_cam + _T
        if transl_cam[..., 2].min() < 0.5:      # If the person is too close to the camera
            transl_cam[..., 2] = transl_cam[..., 2] + (1.0 - transl_cam[..., 2].min())
        
        # If the subject is away from the field of view, put the camera behind
        fov = torch.div(transl_cam[..., :2], transl_cam[..., 2:]).abs()
        if fov.max() > fov_tol:
            t_max = transl_cam[fov.max(1)[0].max(0)[1].item()]
            z_trg = t_max[:2].abs().max(0)[0] / fov_tol
            pad = z_trg - t_max[2]
            transl_cam[..., 2] = transl_cam[..., 2] + pad
        
        c_transl = transl_cam
        c_orinet = _R @ _orient

        c_transl_list.append(c_transl)
        c_orinet_list.append(c_orinet)

    c_transl = torch.stack(c_transl_list)
    c_orinet = torch.stack(c_orinet_list)
    return c_transl, c_orinet