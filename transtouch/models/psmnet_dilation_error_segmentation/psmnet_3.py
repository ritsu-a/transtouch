import math

import torch

from active_zero2.models.psmnet_dilation.psmnet_submodule_3 import *
from active_zero2.utils.confidence import compute_confidence
from active_zero2.utils.reprojection import compute_reproj_loss_patch

from active_zero2.models.psmnet_dilation_error_segmentation.resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from active_zero2.models.psmnet_dilation_error_segmentation.aspp import ASPP, ASPP_Bottleneck

from torch.autograd import Function

class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class hourglass(nn.Module):
    def __init__(self, inplanes, dilation):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, inplanes * 2, kernel_size=3, stride=2, padding=dilation, dilation=dilation),
            nn.BatchNorm3d(inplanes * 2),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            nn.BatchNorm3d(inplanes * 2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, padding=dilation, dilation=dilation),
            nn.BatchNorm3d(inplanes * 2),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            nn.BatchNorm3d(inplanes * 2, inplanes * 2),
            nn.ReLU(inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(
                inplanes * 2,
                inplanes * 2,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                output_padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm3d(inplanes * 2),
        )

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(
                inplanes * 2,
                inplanes,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                output_padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm3d(inplanes),
        )

    def forward(self, x, presqu, postqu):
        out = self.conv1(x)
        pre = self.conv2(out)
        if postqu is not None:
            pre = F.relu(pre + postqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)
        out = self.conv4(out)

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)
        return out, pre, post


class PSMNetDilationErrorSegmentation(nn.Module):
    def __init__(
        self, num_ir: int, min_disp: float, max_disp: float, num_disp: int, set_zero: bool, dilation: int, use_off: bool, alpha: float
    ):
        super(PSMNetDilationErrorSegmentation, self).__init__()
        self.num_ir = num_ir
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.num_disp = num_disp
        self.dilation = dilation
        self.use_off = use_off
        self.alpha = alpha
        assert num_disp % 4 == 0, "Num_disp % 4 should be 0"
        self.num_disp_4 = num_disp // 4
        self.set_zero = set_zero  # set zero for invalid reference image cost volume

        self.disp_list = torch.linspace(min_disp, max_disp, num_disp)
        self.disp_list_4 = torch.linspace(min_disp, max_disp, self.num_disp_4) / 4
        self.disp_regression = DisparityRegression(min_disp, max_disp, num_disp)

        self.feature_extraction = FeatureExtraction(num_ir + 1 if use_off else num_ir)

        self.dres0 = nn.Sequential(
            convbn_3d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.dres1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(32, 32, 3, 1, 1),
        )

        self.dres2 = hourglass(32, dilation)
        self.dres3 = hourglass(32, dilation)
        self.dres4 = hourglass(32, dilation)

        self.classif1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.classif2 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.classif3 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )

        self.error_conv = nn.Sequential(
            convbn(32, 512, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
            convbn(512, 512, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.num_classes = 2
        self.aspp = ASPP(num_classes=self.num_classes) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        
        for param in self.parameters():
            param.requires_grad = False
        for param in self.error_conv.parameters():
            param.requires_grad = True
        for param in self.aspp.parameters():
            param.requires_grad = True


    def forward(self, data_batch):
        img_L, img_R = data_batch["img_l"], data_batch["img_r"]
        if self.use_off:
            img_off_L, img_off_R = data_batch["img_off_l"], data_batch["img_off_r"]
            img_L = torch.cat([img_L, img_off_L], dim=1)
            img_R = torch.cat([img_R, img_off_R], dim=1)
        refimg_feature = self.feature_extraction(img_L)  # [bs, 32, H/4, W/4]
        targetimg_feature = self.feature_extraction(img_R)
        [bs, feature_size, H, W] = refimg_feature.size()

        # error segmentation 
        error_feature1 = refimg_feature.clone().detach()
        error_feature2 = self.error_conv(error_feature1)
        error_output = self.aspp(error_feature2)
        error_segmentation = F.interpolate(
            error_output,
            (4 * H, 4 * W),
            mode="bilinear",
            align_corners=False,
        )

        # # domain classifier
        # domain_feature = self.GRL.apply(refimg_feature, self.alpha)
        # domain_cost1 = self.domain_conv0(domain_feature)
        # domain_cost2 = self.domain_conv1(domain_cost1) + domain_cost1
        # domain_confidence, domain_pre, domain_post = self.domain_hourglass(domain_cost2, None, None)
        # domain_confidence = domain_confidence + domain_cost2
        # domain_out1 = self.domain_classif(domain_confidence)
        # domain_out = F.interpolate(
        #     domain_out1,
        #     (4 * H, 4 * W),
        #     mode="bilinear",
        #     align_corners=False,
        # )

        disp_list = self.disp_list.to(refimg_feature.device)
        disp_list_4 = self.disp_list_4.to(refimg_feature.device)


        # Cost Volume
        
        # Original coordinates of pixels
        x_base = (
            torch.linspace(0, 1, W, dtype=refimg_feature.dtype, device=refimg_feature.device)
            .view(1, 1, W, 1)
            .expand(bs, H, W, self.num_disp_4)
        )
        y_base = (
            torch.linspace(0, 1, H, dtype=refimg_feature.dtype, device=refimg_feature.device)
            .view(1, H, 1, 1)
            .expand(bs, H, W, self.num_disp_4)
        )
        disp_grid = (disp_list_4 / (W - 1)).view(1, 1, 1, self.num_disp_4).expand(bs, H, W, self.num_disp_4)
        target_grids = torch.stack((x_base - disp_grid, y_base), dim=-1).view(bs, H, W * self.num_disp_4, 2)
        target_cost_volume = F.grid_sample(
            targetimg_feature, 2 * target_grids - 1, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        target_cost_volume = target_cost_volume.view(bs, feature_size, H, W, self.num_disp_4).permute(0, 1, 4, 2, 3)
        ref_cost_volume = refimg_feature.unsqueeze(2).expand(bs, feature_size, self.num_disp_4, H, W)
        if self.set_zero:
            # set invalid area to zero
            valid_mask = (x_base > disp_grid).permute(0, 3, 1, 2).unsqueeze(1)
            ref_cost_volume = ref_cost_volume * valid_mask

        cost = torch.cat((ref_cost_volume, target_cost_volume), dim=1)

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0
        
        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        ### delete self.training

        cost1 = F.interpolate(
            cost1,
            (self.num_disp, 4 * H, 4 * W),
            mode="trilinear",
            align_corners=False,
        )
        cost2 = F.interpolate(
            cost2,
            (self.num_disp, 4 * H, 4 * W),
            mode="trilinear",
            align_corners=False,
        )

        cost1 = torch.squeeze(cost1, 1)
        pred1 = F.softmax(cost1, dim=1)
        pred1 = self.disp_regression(pred1)

        cost2 = torch.squeeze(cost2, 1)
        pred2 = F.softmax(cost2, dim=1)
        pred2 = self.disp_regression(pred2)

        ### 

        prob_volume = F.softmax(torch.squeeze(cost3, 1), 1)
        cost3 = F.interpolate(cost3, (self.num_disp, 4 * H, 4 * W), mode="trilinear", align_corners=False)
        cost3 = torch.squeeze(cost3, 1)
        prob_cost3 = F.softmax(cost3, dim=1)

        # For your information: This formulation 'softmax(c)' learned "similarity"
        # while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        # However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = self.disp_regression(prob_cost3)
        pred_norm = (pred3 - self.min_disp) / (self.max_disp - self.min_disp)
        conf_map = compute_confidence(pred_norm, prob_cost3)
        if self.training:
            pred_dict = {
                "pred1": pred1,
                "pred2": pred2,
                "pred3": pred3,
                "conf_map": conf_map,
                "feature_l": refimg_feature,
                "feature_r": targetimg_feature,
                "error_segmentation": error_segmentation,
            }
        else:
            pred_dict = {
                "pred1": pred1,
                "pred2": pred2,
                "pred3": pred3,
                "prob_volume": prob_volume,
                "conf_map": conf_map,
                "feature_l": refimg_feature,
                "feature_r": targetimg_feature,
                "error_segmentation": error_segmentation,
            }
        # TODO: delete the uninterpolated domain out 1, feature

        return pred_dict

    def compute_disp_loss(self, data_batch, pred_dict):
        disp_gt = data_batch["img_disp_l"]
        # Get stereo loss on sim
        # Note in training we do not exclude bg
        mask = (disp_gt < self.max_disp) * (disp_gt > self.min_disp)
        mask.detach()
        loss_disp = 0.0
        for pred_name, loss_weight in zip(["pred1", "pred2", "pred3"], [0.5, 0.7, 1.0]):
            if pred_name in pred_dict:
                loss_disp += loss_weight * F.smooth_l1_loss(pred_dict[pred_name][mask], disp_gt[mask], reduction="mean")

        return loss_disp

    def compute_reproj_loss(self, data_batch, pred_dict, use_mask: bool, patch_size: int, only_last_pred: bool):
        if use_mask:
            disp_gt = data_batch["img_disp_l"]
            # Get stereo loss on sim
            # Note in training we do not exclude bg
            mask = (disp_gt < self.max_disp) * (disp_gt > self.min_disp)
            mask.detach()
        else:
            mask = None
        if only_last_pred:
            loss_reproj = compute_reproj_loss_patch(
                data_batch["img_pattern_l"],
                data_batch["img_pattern_r"],
                pred_disp_l=pred_dict["pred3"],
                mask=mask,
                ps=patch_size,
            )

            return loss_reproj
        else:
            loss_reproj = 0.0
            for pred_name, loss_weight in zip(["pred1", "pred2", "pred3"], [0.5, 0.7, 1.0]):
                if pred_name in pred_dict:
                    loss_reproj += loss_weight * compute_reproj_loss_patch(
                        data_batch["img_pattern_l"],
                        data_batch["img_pattern_r"],
                        pred_disp_l=pred_dict[pred_name],
                        mask=mask,
                        ps=patch_size,
                    )
            return loss_reproj

    def compute_confidence_loss(self, data_batch, pred_dict, domain):

        
        if domain == "sim":
            confidence_gt = torch.zeros_like(pred_dict["domain_confidence"])
            # disp_gt = data_batch["img_disp_l"]
            # mask = (disp_gt < self.max_disp) * (disp_gt > self.min_disp)
            # mask.detach()
            # loss_confidence = torch.nn.BCEWithLogitsLoss()(pred_dict["domain_confidence"][mask], confidence_gt[mask])
            # TODO: recover mask
            loss_confidence = torch.nn.BCEWithLogitsLoss()(pred_dict["domain_confidence"], confidence_gt)


        elif domain == "real":
            confidence_gt = torch.ones_like(pred_dict["domain_confidence"])
            loss_confidence = torch.nn.BCEWithLogitsLoss()(pred_dict["domain_confidence"], confidence_gt)
            
        return loss_confidence


if __name__ == "__main__":
    model = PSMNetDilation(min_disp=12, max_disp=96, num_disp=128, set_zero=False, dilation=3)
    model = model.cuda()
    model.eval()

    data_batch = {
        "img_l": torch.rand(1, 1, 256, 512).cuda(),
        "img_r": torch.rand(1, 1, 256, 512).cuda(),
    }
    pred = model(data_batch)

    for k, v in pred.items():
        print(k, v.shape)
