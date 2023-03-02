import cv2
import torch
from path import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton

from active_zero2.utils.reprojection import compute_reproj_loss_patch_points


def main():
    data_dir = Path("/share/datasets/chenrui/real_data_v10_temp/")
    gt_dir = Path("/share/datasets/chenrui/real_data_v10/")
    scene_id = "0-300104-12"
    scene_dir = data_dir / scene_id
    scene_gt_dir = gt_dir / scene_id

    pattern_patch_size = 11
    reproj_patch_size = 11
    threshold = 0.005

    irL_img = cv2.imread(scene_dir / "1024_irL_real_360.png")
    irL_img = cv2.resize(irL_img, (960, 540), interpolation=cv2.INTER_LANCZOS4)
    irR_img = cv2.imread(scene_dir / "1024_irR_real_360.png")
    irR_img = cv2.resize(irR_img, (960, 540), interpolation=cv2.INTER_LANCZOS4)
    disp_gt = np.load(scene_gt_dir / "disp_gt.npy")

    binl = cv2.imread(scene_dir / f"1024_irL_real_bin_ps{pattern_patch_size}_t{threshold}.png", 0)
    binr = cv2.imread(scene_dir / f"1024_irR_real_bin_ps{pattern_patch_size}_t{threshold}.png", 0)
    bin_img = np.concatenate([binl, binr])
    bin_img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB)
    binl = (binl.astype(float)) / 255.0
    binr = (binr.astype(float)) / 255.0
    binl_torch = torch.from_numpy(binl).float().unsqueeze(0).unsqueeze(0)
    binr_torch = torch.from_numpy(binr).float().unsqueeze(0).unsqueeze(0)

    lcnl = cv2.imread(scene_dir / f"1024_irL_real_lcn_ps{pattern_patch_size}.png", 0)
    lcnr = cv2.imread(scene_dir / f"1024_irR_real_lcn_ps{pattern_patch_size}.png", 0)
    lcn_img = np.concatenate([lcnl, lcnr])
    lcn_img = cv2.cvtColor(lcn_img, cv2.COLOR_GRAY2RGB)
    lcnl = (lcnl.astype(float)) / 255.0
    lcnr = (lcnr.astype(float)) / 255.0
    lcnl_torch = torch.from_numpy(lcnl).float().unsqueeze(0).unsqueeze(0)
    lcnr_torch = torch.from_numpy(lcnr).float().unsqueeze(0).unsqueeze(0)

    templ = cv2.imread(scene_dir / f"1024_irL_real_temporal_ps{pattern_patch_size}_t{threshold}.png", 0)
    tempr = cv2.imread(scene_dir / f"1024_irR_real_temporal_ps{pattern_patch_size}_t{threshold}.png", 0)
    temp_img = np.concatenate([templ, tempr])
    temp_img = cv2.cvtColor(temp_img, cv2.COLOR_GRAY2RGB)
    templ = (templ.astype(float)) / 255.0
    tempr = (tempr.astype(float)) / 255.0
    templ_torch = torch.from_numpy(templ).float().unsqueeze(0).unsqueeze(0)
    tempr_torch = torch.from_numpy(tempr).float().unsqueeze(0).unsqueeze(0)

    # templcnl = cv2.imread(scene_dir / f"1024_irL_real_templcn2_ps{pattern_patch_size}_t0.2.png", 0)
    # templcnr = cv2.imread(scene_dir / f"1024_irR_real_templcn2_ps{pattern_patch_size}_t0.2.png", 0)
    # templcn_img = np.concatenate([templcnl, templcnr])
    # templcn_img = cv2.cvtColor(templcn_img, cv2.COLOR_GRAY2RGB)
    # templcnl = (templcnl.astype(float)) / 255.0
    # templcnr = (templcnr.astype(float)) / 255.0
    # templcnl_torch = torch.from_numpy(templcnl).float().unsqueeze(0).unsqueeze(0)
    # templcnr_torch = torch.from_numpy(templcnr).float().unsqueeze(0).unsqueeze(0)

    fig, axs = plt.subplots(2, 3, figsize=(32, 10))
    axs[0, 0].imshow(irL_img)
    axs[0, 1].imshow(irR_img)
    axs[1, 0].imshow(bin_img)
    axs[1, 1].imshow(lcn_img)
    axs[1, 2].imshow(temp_img)
    # axs[1, 3].imshow(templcn_img)

    def on_click(event):
        if event.inaxes == axs[0, 0]:
            x, y = int(event.xdata), int(event.ydata)
            print(x, y)
            disp_xy = disp_gt[y, x]
            irL_img2 = irL_img.copy()
            irR_img2 = irR_img.copy()
            bin_img2 = bin_img.copy()
            lcn_img2 = lcn_img.copy()
            temp_img2 = temp_img.copy()
            # templcn_img2 = templcn_img.copy()
            cv2.circle(irL_img2, (x, y), 15, (255, 0, 0), 5)
            cv2.line(irR_img2, (0, y), (960, y), (255, 0, 0), 2)
            axs[0, 0].imshow(irL_img2)
            axs[0, 1].imshow(irR_img2)

            cv2.circle(bin_img2, (x, y), 15, (255, 0, 0), 5)
            cv2.circle(lcn_img2, (x, y), 15, (255, 0, 0), 5)
            cv2.circle(temp_img2, (x, y), 15, (255, 0, 0), 5)
            # cv2.circle(templcn_img2, (x, y), 15, (255, 0, 0), 5)
            cv2.line(bin_img2, (0, y + 540), (960, y + 540), (255, 0, 0), 5)
            cv2.line(lcn_img2, (0, y + 540), (960, y + 540), (255, 0, 0), 5)
            cv2.line(temp_img2, (0, y + 540), (960, y + 540), (255, 0, 0), 5)
            # cv2.line(templcn_img2, (0, y + 540), (960, y + 540), (255, 0, 0), 5)
            axs[1, 0].imshow(bin_img2)
            axs[1, 1].imshow(lcn_img2)
            axs[1, 2].imshow(temp_img2)
            # axs[1, 3].imshow(templcn_img2)

            axs[0, 2].clear()
            # compute reprojection loss
            pts = torch.tensor([x, y]).view(1, 2, 1).expand(1, 2, 960)
            disp = x - torch.arange(960)
            disp = disp.view(1, 1, 960)
            bin_loss = compute_reproj_loss_patch_points(
                binl_torch, binr_torch, pts, disp, 540, 960, ps=reproj_patch_size
            )
            bin_loss = torch.mean(bin_loss, dim=(0, 1, 2)).numpy()
            axs[0, 2].plot(bin_loss, "b", label="bin")
            lcn_loss = compute_reproj_loss_patch_points(
                lcnl_torch, lcnr_torch, pts, disp, 540, 960, ps=reproj_patch_size
            )
            lcn_loss = torch.mean(lcn_loss, dim=(0, 1, 2)).numpy()
            axs[0, 2].plot(lcn_loss, "g", label="lcn")
            temp_loss = compute_reproj_loss_patch_points(
                templ_torch, tempr_torch, pts, disp, 540, 960, ps=reproj_patch_size
            )
            temp_loss = torch.mean(temp_loss, dim=(0, 1, 2)).numpy()
            axs[0, 2].plot(temp_loss, "r", label="temp")
            # templcn_loss = compute_reproj_loss_patch_points(
            #     templcnl_torch, templcnr_torch, pts, disp, 540, 960, ps=reproj_patch_size
            # )
            # templcn_loss = torch.mean(templcn_loss, dim=(0, 1, 2)).numpy()
            # axs[0, 2].plot(templcn_loss, "c", label="templcn")

            x2 = x - disp_xy
            axs[0, 2].axvline(x2, color="m")

            axs[0, 2].legend()
            fig.canvas.draw()
            fig.canvas.flush_events()
        # elif event.inaxes == axs[0, 1]:
        #     x2, y2 = int(event.xdata), int(event.ydata)
        #     print(x2, y2)
        #     irR_img2 = irR_img.copy()
        #     cv2.circle(irR_img2, (x2, y2), 15, (255, 0, 0), 5)
        #     axs[0, 1].imshow(irR_img2)
        #     if len(axs[0, 2].lines) == 5:
        #         axs[0, 2].lines.pop(-1)
        #     axs[0, 2].axvline(x2, color="m")
        #     fig.canvas.draw()
        #     fig.canvas.flush_events()

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


if __name__ == "__main__":
    main()
