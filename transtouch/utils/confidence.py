import torch


def compute_confidence(disp_pred_norm, prob_volume, margin = 5) -> torch.Tensor:
    disp_pred_norm = disp_pred_norm.cpu()
    prob_volume = prob_volume.cpu()
    B, D, H, W = prob_volume.shape
    grid = torch.linspace(0, 1, D).view(1, D, 1, 1).expand(B, D, H, W)
    grid_new = torch.linspace(0, D, D).view(1, D, 1, 1).expand(B, D, H, W)
    confs = []
    with torch.no_grad():
        disp_pred = disp_pred_norm.expand(B, D, H, W)
        grid_new = (torch.abs(grid_new - disp_pred * D)) ** 2
        confs.append(torch.sum(grid * prob_volume, 1, keepdim=True))
        for m in range(1, margin + 1):
            disp_pred_floor = disp_pred_norm - m / (D - 1)
            disp_pred_floor = torch.clip(disp_pred_floor, 0, 1)
            disp_pred_ceil = disp_pred_norm + m / (D - 1)
            disp_pred_ceil = torch.clip(disp_pred_ceil, 0, 1)
            mask = (grid >= disp_pred_floor) * (grid <= disp_pred_ceil)
            conf = torch.sum(mask * prob_volume, 1, keepdim=True)
            confs.append(conf)

        confs = torch.cat(confs, dim=1)

    return confs.cuda()

def compute_confidence_variance(disp_pred_norm, prob_volume, margin=4) -> torch.Tensor:
    disp_pred_norm = disp_pred_norm.cpu()
    prob_volume = prob_volume.cpu()
    B, D, H, W = prob_volume.shape
    disp_pred = disp_pred_norm.expand(B, D, H, W).to(prob_volume.device)
    grid = torch.linspace(0, D, D).view(1, D, 1, 1).expand(B, D, H, W).to(prob_volume.device)
    with torch.no_grad():
        grid = (torch.abs(grid - disp_pred * D)) ** 2
        confs = torch.sum(grid * prob_volume, 1, keepdim=True)
    return confs.cuda()

def compute_confidence_KL(disp_pred_norm, prob_volume, margin=4) -> torch.Tensor:
    disp_pred_norm = disp_pred_norm.cpu()
    prob_volume = prob_volume.cpu()
    B, D, H, W = prob_volume.shape
    disp_pred = disp_pred_norm.expand(B, D, H, W).to(prob_volume.device)
    grid = torch.linspace(0, D, D).view(1, D, 1, 1).expand(B, D, H, W).to(prob_volume.device)
    with torch.no_grad():
        grid = (torch.abs(grid - disp_pred * D)) ** 2
        variance = torch.sum(grid * prob_volume, 1, keepdim=True)
        grid = torch.exp( - grid / (2 * 0.1))
        grid /= torch.sum(grid, 1, keepdim=True)
        kl = (grid * torch.nan_to_num(torch.log( grid / (prob_volume+1e-10))))
        print("-"*100)
    return torch.sum(kl, 1, keepdim=True).cuda()