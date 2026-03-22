import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_utils import match, log_sum_exp

class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets, mask_targets):
        loc_data, conf_data, landm_data, gaze_data, hdps_data, dist_data, fc_gaze, fc_hdps, fc_landmark, loc_data_pre, conf_data_pre = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))

        targets_concat = torch.concat(targets, 0)
        loss_fc_gaze = F.smooth_l1_loss(fc_gaze, targets_concat[:, 28:31], reduction='sum')
        loss_fc_hdps = F.smooth_l1_loss(fc_hdps, targets_concat[:, 31:34], reduction='sum')
        loss_fc_landmark = F.smooth_l1_loss(fc_landmark, targets_concat[:, 4:28], reduction='sum')
        
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 24)
        gaze_t = torch.Tensor(num, num_priors, 3)
        hdps_t = torch.Tensor(num, num_priors, 3)
        dist_t = torch.Tensor(num, num_priors, 1)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data
            landms = targets[idx][:, 4:28].data
            gazes  = targets[idx][:, 28:31].data
            hdpses = targets[idx][:, 31:34].data
            dists  = targets[idx][:, 34:35].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, landms, gazes, hdpses, dists,
                  loc_t, conf_t, landm_t, gaze_t, hdps_t, dist_t, idx)
        loc_t = loc_t.cuda()
        conf_t = conf_t.cuda()
        landm_t = landm_t.cuda()
        gaze_t = gaze_t.cuda()
        hdps_t = hdps_t.cuda()
        dist_t = dist_t.cuda()

        zeros = torch.tensor(0).cuda()
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,24]
        pos1 = conf_t > zeros
        num_pos_landm = pos1.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        landm_p = landm_data[pos_idx1].view(-1, 24)
        landm_t = landm_t[pos_idx1].view(-1, 24)
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos = conf_t != 0
        conf_t[pos] = 1
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # gaze Loss (Smooth L1)
        # Shape: [batch,num_priors,3]
        pos = conf_t != 0
        conf_t[pos] = 1
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(gaze_data)
        gaze_p = gaze_data[pos_idx].view(-1, 3)
        gaze_t = gaze_t[pos_idx].view(-1, 3)
        loss_gaze = F.smooth_l1_loss(gaze_p, gaze_t, reduction='sum')
        
        # hdps Loss (Smooth L1)
        # Shape: [batch,num_priors,3]
        hdps_p = hdps_data[pos_idx].view(-1, 3)
        hdps_t = hdps_t[pos_idx].view(-1, 3)
        loss_hdps = F.smooth_l1_loss(hdps_p, hdps_t, reduction='sum')

        # distance Loss (Smooth L1)
        # Shape: [batch,num_priors,1]
        pos = conf_t != 0
        conf_t[pos] = 1
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(dist_data)
        dist_p = dist_data[pos_idx].view(-1, 1)
        dist_t = dist_t[pos_idx].view(-1, 1)
        loss_dist = F.smooth_l1_loss(dist_p, dist_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # pre

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos = conf_t != 0
        conf_t[pos] = 1
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data_pre)
        loc_p = loc_data_pre[pos_idx].view(-1, 4)
        # loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l_pre = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data_pre.view(-1, self.num_classes)
        loss_c_pre = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c_pre[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c_pre = loss_c_pre.view(num, -1)
        _, loss_idx = loss_c_pre.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data_pre)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data_pre)
        conf_p = conf_data_pre[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c_pre = F.cross_entropy(conf_p, targets_weighted, reduction='sum')



        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1
        loss_gaze /= N
        loss_hdps /= N
        loss_dist /= N

        loss_fc_landmark /= N
        loss_fc_gaze /= N
        loss_fc_hdps /= N
        loss_l_pre /= N
        loss_c_pre /= N

        return loss_l, loss_c, loss_landm, loss_gaze, loss_hdps, loss_dist, loss_fc_landmark, loss_fc_gaze, loss_fc_hdps, loss_l_pre, loss_c_pre
