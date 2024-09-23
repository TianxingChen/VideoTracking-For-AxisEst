import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .pspnet import PSPNet
from .rotation_utils import Ortho6d2Mat

class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class StereoPoseNet(nn.Module):
    def __init__(self, n_cat=1, img_arc = 'ResNet34', use_pc = False, pc_arc = None, stereo_fusion = True):
        super(StereoPoseNet, self).__init__()
        self.n_cat = n_cat
        self.img_arc = img_arc
        self.pc_arc = pc_arc
        self.use_pc = use_pc
        self.stereo_fusion = stereo_fusion
        self.img_extractor = PSPNet(bins=(1, 2, 3, 6), backend='resnet34')
        
        self.instance_color = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.ReLU(inplace=True)
        )

        self.volume_conv = nn.Sequential(
            Conv3d(32, 16, 1, 1),
            Conv3d(16, 8, 1, 1),
            Conv3d(8, 1, 1, 1)
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(24, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1)
        )

        self.nocs_head = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 3, 1),
            nn.Tanh()
        )

        self.nocs_pts_mlp = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU()
        )

        self.pose_mlp1 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.ReLU()
        )

        self.pose_mlp2 = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.rotation_estimator = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        self.translation_estimator = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        self.size_estimator = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
    
    def homo_warping(self, src_fea, src_proj, ref_proj, depth_values):
        # src_fea: [B, C, H, W]
        # src_proj: [B, 4, 4]
        # ref_proj: [B, 4, 4]
        # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
        # out: [B, C, Ndepth, H, W]

        batch, channels = src_fea.shape[0], src_fea.shape[1]
        num_depth = depth_values.shape[1]
        height, width = src_fea.shape[2], src_fea.shape[3]

        with torch.no_grad():
            proj = torch.matmul(src_proj, torch.inverse(ref_proj))
            rot = proj[:, :3, :3]  # [B,3,3]
            trans = proj[:, :3, 3:4]  # [B,3,1]

            y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                                torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
            y, x = y.contiguous(), x.contiguous()
            y, x = y.view(height * width), x.view(height * width)

            xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
            xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
            
            rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
            rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth, -1)  # [B, 3, Ndepth, H*W]
            proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]

            proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
            proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
            proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
            proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
            grid = proj_xy
        warped_src_fea = F.grid_sample(src_fea,
                                    grid.view(batch, num_depth * height,
                                        width, 2), mode='bilinear', padding_mode='zeros')
        warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

        return warped_src_fea

    def forward(self, view1_img, view1_choose, view2_img, view2_choose, view1_proj, view2_proj, depth_values):
        assert view1_choose.shape == view2_choose.shape
        bs, n_pts = view1_choose.size()
        num_depth = depth_values.shape[1]

        view1_feat_img = self.img_extractor(view1_img)
        view2_feat_img = self.img_extractor(view2_img)

        warped_view2_features = self.homo_warping(view2_feat_img, view2_proj, view1_proj, depth_values) # (B, C, N, H, W)
        warped_view1_features = self.homo_warping(view1_feat_img, view1_proj, view2_proj, depth_values) # (B, C, N, H, W)

        fused_view1_features = view1_feat_img.unsqueeze(2).repeat(1, 1, num_depth, 1, 1) + warped_view2_features        
        fused_view2_features = view2_feat_img.unsqueeze(2).repeat(1, 1, num_depth, 1, 1) + warped_view1_features

        fused_view2_features = self.volume_conv(fused_view2_features).squeeze(1)
        fused_view1_features = self.volume_conv(fused_view1_features).squeeze(1)

        view1_feat_img = F.relu(view1_feat_img + self.fuse_conv(fused_view1_features))
        view2_feat_img = F.relu(view2_feat_img + self.fuse_conv(fused_view2_features))

        di = view1_feat_img.size()[1]
        view1_emb = view1_feat_img.view(bs, di, -1)
        view1_choose = view1_choose.unsqueeze(1).repeat(1, di, 1)
        view1_feat = torch.gather(view1_emb, 2, view1_choose).contiguous()
        view1_feat = self.instance_color(view1_feat)

        view2_emb = view2_feat_img.view(bs, di, -1)
        view2_choose = view2_choose.unsqueeze(1).repeat(1, di, 1)
        view2_feat = torch.gather(view2_emb, 2, view2_choose).contiguous()
        view2_feat = self.instance_color(view2_feat)

        view1_nocs = self.nocs_head(view1_feat)
        view2_nocs = self.nocs_head(view2_feat)

        view1_nocs_pts_feat = self.nocs_pts_mlp(view1_nocs)
        view2_nocs_pts_feat = self.nocs_pts_mlp(view2_nocs)

        view1_pose_feat = torch.cat((view1_feat, view1_nocs_pts_feat), dim=1)
        view2_pose_feat = torch.cat((view2_feat, view2_nocs_pts_feat), dim=1)

        view1_pose_feat = self.pose_mlp1(view1_pose_feat)
        view1_pose_global = torch.mean(view1_pose_feat, 2, keepdim=True)
        view1_pose_feat1 = torch.cat([view1_pose_feat, view1_pose_global.expand_as(view1_pose_feat)], 1)
        view1_pose_feat2 = self.pose_mlp2(view1_pose_feat1).squeeze(2)
        view1_r = self.rotation_estimator(view1_pose_feat2)
        view1_r = Ortho6d2Mat(view1_r[:, :3].contiguous(), view1_r[:, 3:].contiguous()).view(-1,3,3)
        view1_t = self.translation_estimator(view1_pose_feat2)
        view1_s = self.size_estimator(view1_pose_feat2)

        view2_pose_feat = self.pose_mlp1(view2_pose_feat)
        view2_pose_global = torch.mean(view2_pose_feat, 2, keepdim=True)
        view2_pose_feat1 = torch.cat([view2_pose_feat, view2_pose_global.expand_as(view2_pose_feat)], 1)
        view2_pose_feat2 = self.pose_mlp2(view2_pose_feat1).squeeze(2)
        view2_r = self.rotation_estimator(view2_pose_feat2)
        view2_r = Ortho6d2Mat(view2_r[:, :3].contiguous(), view2_r[:, 3:].contiguous()).view(-1,3,3)
        view2_t = self.translation_estimator(view2_pose_feat2)
        view2_s = self.size_estimator(view2_pose_feat2)

        return {'view1_nocs': view1_nocs.permute(0, 2, 1).contiguous(), 'view1_r': view1_r, 'view1_t': view1_t, 'view1_s': view1_s, \
            'view2_nocs': view2_nocs.permute(0, 2, 1).contiguous(), 'view2_r': view2_r, 'view2_t': view2_t, 'view2_s': view2_s}

class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostRegNet, self).__init__()
        self.conv0 = Conv3d(in_channels, base_channels, padding=1)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x

def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        # print("regression dim <= 2")
        depth_values = depth_values.view(*depth_values.shape, 1)
    depth = torch.sum(p * depth_values, 1)

    return depth

class StereoPoseNet_with_depth(nn.Module):
    def __init__(self, n_cat=1, nv_pts = 1024, img_arc = 'ResNet34', stereo_fusion = True, regress_pose = False):
        super(StereoPoseNet_with_depth, self).__init__()
        self.n_cat = n_cat
        self.img_arc = img_arc
        self.stereo_fusion = stereo_fusion
        self.regress_pose = regress_pose

        # image feature extractor
        if img_arc == 'ResNet18':
            self.img_extractor = PSPNet(bins=(1, 2, 3, 6), backend='resnet18')
        elif img_arc == 'ResNet34':
            self.img_extractor = PSPNet(bins=(1, 2, 3, 6), backend='resnet34')
        else:
            raise NotImplementedError
        
        self.instance_color = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.ReLU(inplace=True)
        )

        self.cost_regularization = CostRegNet(in_channels=32, base_channels=8)

        self.nocs_head = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 3, 1),
            nn.Tanh()
        )

        if self.regress_pose:
            self.nocs_pts_mlp = nn.Sequential(
                nn.Conv1d(3, 32, 1),
                nn.ReLU(),
                nn.Conv1d(32, 64, 1),
                nn.ReLU()
            )

            self.pose_mlp1 = nn.Sequential(
                nn.Conv1d(96, 128, 1),
                nn.ReLU(),
                nn.Conv1d(128, 128, 1),
                nn.ReLU()
            )

            self.pose_mlp2 = nn.Sequential(
                nn.Conv1d(256, 256, 1),
                nn.ReLU(),
                nn.Conv1d(256, 256, 1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )

            self.rotation_estimator = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 6)
            )
            self.translation_estimator = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 3)
            )
            self.size_estimator = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 3)
            )
    
    def homo_warping(self, src_fea, src_proj, ref_proj, depth_values):
        # src_fea: [B, C, H, W]
        # src_proj: [B, 4, 4]
        # ref_proj: [B, 4, 4]
        # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
        # out: [B, C, Ndepth, H, W]

        batch, channels = src_fea.shape[0], src_fea.shape[1]
        num_depth = depth_values.shape[1]
        height, width = src_fea.shape[2], src_fea.shape[3]

        with torch.no_grad():
            proj = torch.matmul(src_proj, torch.inverse(ref_proj))
            rot = proj[:, :3, :3]  # [B,3,3]
            trans = proj[:, :3, 3:4]  # [B,3,1]

            y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                                torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
            y, x = y.contiguous(), x.contiguous()
            y, x = y.view(height * width), x.view(height * width)

            xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
            xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
            
            rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
            rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth, -1)  # [B, 3, Ndepth, H*W]
            proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]

            proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
            proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
            proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
            proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
            grid = proj_xy
        warped_src_fea = F.grid_sample(src_fea,
                                    grid.view(batch, num_depth * height,
                                        width, 2), mode='bilinear', padding_mode='zeros')
        warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

        return warped_src_fea
    
    def forward(self, view1_img, view1_choose, view2_img, view2_choose, view1_proj, view2_proj, depth_values):
        assert view1_choose.shape == view2_choose.shape
        bs, n_pts = view1_choose.size()
        num_depth = depth_values.shape[1]

        view1_feat_img = self.img_extractor(view1_img)
        view2_feat_img = self.img_extractor(view2_img)

        warped_view2_features = self.homo_warping(view2_feat_img, view2_proj, view1_proj, depth_values) # (B, C, N, H, W)
        warped_view1_features = self.homo_warping(view1_feat_img, view1_proj, view2_proj, depth_values) # (B, C, N, H, W)

        fused_view1_features = view1_feat_img.unsqueeze(2).repeat(1, 1, num_depth, 1, 1) + warped_view2_features        
        fused_view2_features = view2_feat_img.unsqueeze(2).repeat(1, 1, num_depth, 1, 1) + warped_view1_features

        di = view1_feat_img.size()[1]
        view1_emb = view1_feat_img.view(bs, di, -1)
        view1_nocs_choose = view1_choose.unsqueeze(1).repeat(1, di, 1)
        view1_nocs_feat = torch.gather(view1_emb, 2, view1_nocs_choose).contiguous()
        view1_nocs_feat = self.instance_color(view1_nocs_feat)

        view2_emb = view2_feat_img.view(bs, di, -1)
        view2_nocs_choose = view2_choose.unsqueeze(1).repeat(1, di, 1)
        view2_nocs_feat = torch.gather(view2_emb, 2, view2_nocs_choose).contiguous()
        view2_nocs_feat = self.instance_color(view2_nocs_feat)

        view1_nocs = self.nocs_head(view1_nocs_feat)
        view2_nocs = self.nocs_head(view2_nocs_feat)

        view1_cost_reg = self.cost_regularization(fused_view1_features)
        view2_cost_reg = self.cost_regularization(fused_view2_features)

        prob_view1_volume_pre = view1_cost_reg.squeeze(1)
        di = prob_view1_volume_pre.size()[1]
        prob_view1_volume_pre = prob_view1_volume_pre.view(bs, di, -1)
        view1_depth_choose = view1_choose.unsqueeze(1).repeat(1, di, 1)
        prob_view1_volume_pre = torch.gather(prob_view1_volume_pre, 2, view1_depth_choose).contiguous()
        prob_view1_volume = F.softmax(prob_view1_volume_pre, dim=1)
        view1_depth = depth_regression(prob_view1_volume, depth_values=depth_values)

        if self.regress_pose:
            # depth-guided fusion
            bs, dim, n_depth, h, w = fused_view1_features.shape
            fused_view1_features = fused_view1_features.view(bs, -1, h, w)
            di = fused_view1_features.size()[1]
            fused_view1_features = fused_view1_features.view(bs, di, -1)
            view1_depth_choose = view1_choose.unsqueeze(1).repeat(1, di, 1)
            fused_view1_features = torch.gather(fused_view1_features, 2, view1_depth_choose).contiguous().view(bs, dim, n_depth, -1)
            fused_view1_features = torch.sum(fused_view1_features * (prob_view1_volume.unsqueeze(1).repeat(1, dim, 1, 1)), dim=2)

        prob_view2_volume_pre = view2_cost_reg.squeeze(1)
        di = prob_view2_volume_pre.size()[1]
        prob_view2_volume_pre = prob_view2_volume_pre.view(bs, di, -1)
        view2_depth_choose = view2_choose.unsqueeze(1).repeat(1, di, 1)
        prob_view2_volume_pre = torch.gather(prob_view2_volume_pre, 2, view2_depth_choose).contiguous()
        prob_view2_volume = F.softmax(prob_view2_volume_pre, dim=1)
        view2_depth = depth_regression(prob_view2_volume, depth_values=depth_values)

        if self.regress_pose:
            # depth-guided fusion
            bs, dim, n_depth, h, w = fused_view2_features.shape
            fused_view2_features = fused_view2_features.view(bs, -1, h, w)
            di = fused_view2_features.size()[1]
            fused_view2_features = fused_view2_features.view(bs, di, -1)
            view2_depth_choose = view2_choose.unsqueeze(1).repeat(1, di, 1)
            fused_view2_features = torch.gather(fused_view2_features, 2, view2_depth_choose).contiguous().view(bs, dim, n_depth, -1)
            fused_view2_features = torch.sum(fused_view2_features * (prob_view2_volume.unsqueeze(1).repeat(1, dim, 1, 1)), dim=2)
        
        if self.regress_pose:
            view1_nocs_pts_feat = self.nocs_pts_mlp(view1_nocs)
            view2_nocs_pts_feat = self.nocs_pts_mlp(view2_nocs)

            view1_pose_feat = torch.cat((fused_view1_features, view1_nocs_pts_feat), dim=1)
            view2_pose_feat = torch.cat((fused_view2_features, view2_nocs_pts_feat), dim=1)

            view1_pose_feat = self.pose_mlp1(view1_pose_feat)
            view1_pose_global = torch.mean(view1_pose_feat, 2, keepdim=True)
            view1_pose_feat1 = torch.cat([view1_pose_feat, view1_pose_global.expand_as(view1_pose_feat)], 1)
            view1_pose_feat2 = self.pose_mlp2(view1_pose_feat1).squeeze(2)
            view1_r = self.rotation_estimator(view1_pose_feat2)
            view1_r = Ortho6d2Mat(view1_r[:, :3].contiguous(), view1_r[:, 3:].contiguous()).view(-1,3,3)
            view1_t = self.translation_estimator(view1_pose_feat2)
            view1_s = self.size_estimator(view1_pose_feat2)

            view2_pose_feat = self.pose_mlp1(view2_pose_feat)
            view2_pose_global = torch.mean(view2_pose_feat, 2, keepdim=True)
            view2_pose_feat1 = torch.cat([view2_pose_feat, view2_pose_global.expand_as(view2_pose_feat)], 1)
            view2_pose_feat2 = self.pose_mlp2(view2_pose_feat1).squeeze(2)
            view2_r = self.rotation_estimator(view2_pose_feat2)
            view2_r = Ortho6d2Mat(view2_r[:, :3].contiguous(), view2_r[:, 3:].contiguous()).view(-1,3,3)
            view2_t = self.translation_estimator(view2_pose_feat2)
            view2_s = self.size_estimator(view2_pose_feat2)

        if self.regress_pose:
            return {'view1_nocs': view1_nocs.permute(0, 2, 1).contiguous(), \
                'view2_nocs': view2_nocs.permute(0, 2, 1).contiguous(), \
                'view1_depth': view1_depth, 'view2_depth': view2_depth, \
                'view1_r': view1_r, 'view1_t': view1_t, 'view1_s': view1_s, \
                'view2_r': view2_r, 'view2_t': view2_t, 'view2_s': view2_s}
        else:
            return {'view1_nocs': view1_nocs.permute(0, 2, 1).contiguous(), \
            'view2_nocs': view2_nocs.permute(0, 2, 1).contiguous(), \
            'view1_depth': view1_depth, 'view2_depth': view2_depth}