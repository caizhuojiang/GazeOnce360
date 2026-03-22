import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torch.nn.functional as F
import torch.nn.init as init

from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import SSH as SSH
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode
from utils.nms.py_cpu_nms import py_cpu_nms

def decode_bbox(loc, conf, priors, variances, confidence_threshold=0.9, nms_threshold=0.4):
    boxes = decode(loc.data.squeeze(0), priors, variances)
    boxes *= 2048
    conf = F.softmax(conf, dim=-1)
    boxes = boxes
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1]
    # order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    dets[:, :4] /= 2048
    
    return dets

def get_hr_imgs(img_raw, bbox, width, height, device, res_hr=2048):
    assert img_raw.shape[0] == 1, "Image batch size must be 1"
    hr_imgs = []
    hr_offsets = []
    for ti, t in enumerate(bbox):
        t[0], t[2] = t[0] * width, t[2] * width
        t[1], t[3] = t[1] * height, t[3] * height
        cx = int((t[0] + t[2]) / 2)
        cy = int((t[1] + t[3]) / 2)
        w = int(t[2] - t[0])
        h = int(t[3] - t[1])
        mwh = max(w, h)
        x1 = int(cx - mwh / 2)
        y1 = int(cy - mwh / 2)
        x2 = int(cx + mwh / 2)
        y2 = int(cy + mwh / 2)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        hr_img = img_raw[:, :, y1:y2, x1:x2]
        hr_img = F.interpolate(hr_img, size=(res_hr//8, res_hr//8), mode='bilinear', align_corners=False)
        hr_imgs.append(hr_img)
        hr_offset = [x1, y1, x2, y2]
        hr_offsets.append(hr_offset)
    hr_imgs = torch.cat(hr_imgs, dim=0)
    hr_offsets = torch.tensor(hr_offsets)
    hr_imgs = torch.concat((hr_imgs, torch.zeros((1, 3, res_hr//8, res_hr//8)).to(device)), dim=0).to(torch.float32)
    hr_offsets = torch.concat((hr_offsets, torch.zeros((1, 4))), dim=0).to(torch.float32)

    hr_imgs = hr_imgs.unsqueeze(0)
    hr_offsets = torch.concat((torch.zeros((len(hr_offsets), 1)), hr_offsets/res_hr), 1).unsqueeze(0)

    hr_imgs = hr_imgs.to(device)
    hr_offsets = hr_offsets.to(device)

    return hr_imgs, hr_offsets

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, max_height: int = 1000, max_width: int = 1000):
        """
        Args:
            d_model: Embedding dimension. Must be even because half is used for rows and half for columns.
            max_height: Maximum image height.
            max_width: Maximum image width.
        """
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for 2D positional encoding"
        
        # Generate positional encodings for rows and columns separately.
        pe_height = self._generate_pe(d_model // 2, max_height)  # (max_height, d_model//2)
        pe_width = self._generate_pe(d_model // 2, max_width)    # (max_width, d_model//2)
        
        # Register as buffers so they are saved/loaded with the model but are not trainable.
        self.register_buffer('pe_height', pe_height)
        self.register_buffer('pe_width', pe_width)

        pe_height = pe_height.unsqueeze(1)  # (height, 1, d_model//2)
        pe_width = pe_width.unsqueeze(0)    # (1, width, d_model//2)
        pe_2d = torch.cat([pe_height.expand(-1, max_width, -1), 
                           pe_width.expand(max_height, -1, -1)], dim=-1)
        self.register_buffer('pe_2d', pe_2d)  # (max_height, max_width, d_model)
        pe = pe_2d.view(max_height * max_width, d_model)  # (max_height * max_width, d_model)
        self.register_buffer('pe', pe)  # (max_height * max_width, d_model)
    
    def _generate_pe(self, d_model_half: int, max_len: int) -> torch.Tensor:
        """Generate 1D sinusoidal/cosine positional encodings (internal helper)."""
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model_half, 2) * (-math.log(10000.0) / d_model_half))
        pe = torch.zeros(max_len, d_model_half)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (max_len, d_model_half)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor with shape (batch_size, height, width, d_model).
        Returns:
            Tensor with positional encodings added, with the same shape as the input.
        """
        # Add positional encoding to the input (broadcast along batch dimension).
        return x + self.pe_2d.unsqueeze(0)
    
class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.nhead = nhead
        
    def forward(self, query, key, value, q_pos_embed=None, kv_pos_embed=None, attn_mask=None, key_padding_mask=None, need_weights=False):
        """
        query: [L, N, C]
        key/value: [S, N, C]
        """
        if q_pos_embed is not None:
            query = query + q_pos_embed
        if kv_pos_embed is not None:
            key = key + kv_pos_embed
            value = value + kv_pos_embed
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1)
            attn_mask = attn_mask.view(attn_mask.shape[0] * attn_mask.shape[1], attn_mask.shape[2], attn_mask.shape[3])
            
        attn_output, attn_weights = self.multihead_attn(
            query, key, value,
            # key_padding_mask=key_padding_mask, 
            attn_mask=attn_mask,
            need_weights=need_weights
        )
        return self.norm(query + attn_output), attn_weights
    
class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*24,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 24)

class GazeHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(GazeHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*3,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 3)
    
class HdpsHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(HdpsHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*3,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 3)

class DistanceHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(DistanceHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*1,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 1)

class GazeOnce360(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(GazeOnce360,self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])
            enhance_backbone = models.resnet50(pretrained=cfg['pretrain'])
        else:
            raise ValueError("Unsupported backbone network: {}".format(cfg['name']))

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        self.enhance_body = _utils.IntermediateLayerGetter(enhance_backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.GazeHead = self._make_gaze_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.HdpsHead = self._make_hdps_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.DistHead = self._make_distance_head(fpn_num=3, inchannels=cfg['out_channel'])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1000)
        self.FCGazeHead = nn.Linear(1000, 3)
        self.FCHdpsHead = nn.Linear(1000, 3)
        self.FCLandmarkHead = nn.Linear(1000, 24)

        self.q_pos_encoders = nn.ModuleDict({
            str(i): PositionalEncoding2D(cfg['in_channel'], 512 // (2**(i+3)), 512 // (2**(i+3)))
            for i in range(len(cfg['return_layers']))
        })

        self.kv_pos_encoders = nn.ModuleDict({
            str(i): PositionalEncoding2D(cfg['in_channel'] * (2**(i+1)), 2048 // (2**(i+3)), 2048 // (2**(i+3))) 
            for i in range(len(cfg['return_layers']))
        })
        
        self.cross_attentions = nn.ModuleDict({
            str(i): CrossAttentionLayer(cfg['in_channel'])
            for i in range(len(cfg['return_layers']))
        })
        
        self.proj_queries = nn.ModuleDict({
            str(i): nn.Linear(cfg['in_channel'], cfg['in_channel'])
            for i in range(len(cfg['return_layers']))
        })

        self.proj_kvs = nn.ModuleDict({
            str(i): nn.Linear(cfg['in_channel'] * (2**(i+1)), cfg['in_channel'])
            for i in range(len(cfg['return_layers']))
        })
        
        self.proj_outputs = nn.ModuleDict({
            str(i): nn.Sequential(
                nn.Linear(cfg['in_channel'], cfg['in_channel']),
                nn.ReLU()
            )
            for i in range(len(cfg['return_layers']))
        })

        self._init_enhance_weights()

        self.priorbox = PriorBox(cfg, image_size=(cfg['image_size'], cfg['image_size']))
        self.priors = self.priorbox.forward()
        self.priors = self.priors.to(torch.device('cuda:0'))
        self.prior_data = self.priors.data
        self.variances = cfg['variance']

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def _make_gaze_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        gazehead = nn.ModuleList()
        for i in range(fpn_num):
            gazehead.append(GazeHead(inchannels,anchor_num))
        return gazehead
    
    def _make_hdps_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        hdpshead = nn.ModuleList()
        for i in range(fpn_num):
            hdpshead.append(HdpsHead(inchannels,anchor_num))
        return hdpshead

    def _make_distance_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        distancehead = nn.ModuleList()
        for i in range(fpn_num):
            distancehead.append(DistanceHead(inchannels,anchor_num))
        return distancehead

    def _fuse_with_attention(self, main_feats, hr_feats, offsets):
        fused_feats = []

        # 2. Prepare high-res features as key/value [N, C]
        hr_patch = list(hr_feats.values())[2]  # [N,C,h,w]
        hr_flat = hr_patch.mean([2,3])    # Global average pooling [N, C]
        hr_flat = self.proj_kvs[str(2)](hr_flat)
        
        for i, main_feat in enumerate(main_feats):
            B, C, H, W = main_feat.shape

            # 1. Prepare main features as query [B*H*W, C]
            main_flat = main_feat.permute(0,2,3,1)
            main_flat = self.q_pos_encoders[str(i)](main_flat)  # [B, H, W, C]
            main_flat = main_flat.view(B, -1, C)
            query = self.proj_queries[str(i)](main_flat)  # [B, H*W, C]

            attn_mask = torch.ones(B, H*W, offsets.shape[1]).to(hr_flat.device).to(torch.bool)

            si = 0
            key = torch.zeros(offsets.shape[0], offsets.shape[1], hr_flat.shape[1]).to(hr_flat.device)
            key_padding_mask = torch.ones(offsets.shape[0], offsets.shape[1]).to(hr_flat.device).to(torch.bool)
            for j in range(len(offsets)):
                l = (offsets[j][:, 1] > 0).sum()
                key[j, :l] = hr_flat[si:si+l]
                key_padding_mask[j, :l] = False

                # Create attention mask for this batch
                for k in range(len(offsets[j])):
                    if k < l:
                        x_start, y_start, x_end, y_end = offsets[j][k][1:]
                        x_start, x_end = int(x_start * W), int(x_end * W)
                        y_start, y_end = int(y_start * H), int(y_end * H)
                        # Convert bbox to attention mask
                        for y in range(y_start, y_end):
                            for x in range(x_start, x_end):
                                pos = y * W + x
                                attn_mask[j, pos, k] = False
                    else:
                        if k < len(offsets[j])-1:
                            attn_mask[j, 0, k] = False  # Padding mask
                        else:
                            attn_mask[j, :, k] = False  # All positions for the last key
                si += l
            value = key.clone()

            # 4. Cross attention
            attn_output, attn_weights = self.cross_attentions[str(i)](
                query, 
                key, 
                value,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False
            )  # [B, H*W, C]

            # 5. Project output and reshape back to feature map format
            output = self.proj_outputs[str(i)](attn_output)  # [B, H*W, C]
            output = output.reshape(B, H, W, C).permute(0,3,1,2)  # [B, C, H, W]
            
            fused_feats.append(output)
            
        return fused_feats

    def _init_enhance_weights(self):
        """Initialize parameters for the enhancement modules."""
        # 1. Initialize query/key/value matrices in cross-attention layers
        for name, layer in self.cross_attentions.items():
            if hasattr(layer, 'query'):
                init.xavier_uniform_(layer.query.weight)
                init.zeros_(layer.query.bias)
            if hasattr(layer, 'key'):
                init.xavier_uniform_(layer.key.weight)
                init.zeros_(layer.key.bias)
            if hasattr(layer, 'value'):
                init.xavier_uniform_(layer.value.weight)
                init.zeros_(layer.value.bias)
        
        # 2. Initialize projection layers (Linear + ReLU)
        for name, layer in self.proj_queries.items():
            init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                init.zeros_(layer.bias)
        
        for name, layer in self.proj_outputs.items():
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        init.zeros_(m.bias)

    def forward(self, inputs, inputs_hr=None, offsets=None, images_full_hr=None, res_hr=2048):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions_pre = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications_pre = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)

        if inputs_hr is not None:
            inputs_hr = inputs_hr.view(-1, 3, res_hr//8, res_hr//8)
            inputs_hr = inputs_hr[inputs_hr.view(-1, 3*(res_hr//8)*(res_hr//8)).sum(dim=1) != 0]
            out_hr = self.enhance_body(inputs_hr)
            fc_in = list(out_hr.values())[2]
            fc_in = self.avgpool(fc_in).view(fc_in.size(0), -1)
            fc_out = self.fc(fc_in)
            fc_gaze = self.FCGazeHead(fc_out)
            fc_hdps = self.FCHdpsHead(fc_out)
            fc_landmark = self.FCLandmarkHead(fc_out)
        elif images_full_hr is not None:
            bbox = decode_bbox(bbox_regressions_pre, classifications_pre, self.prior_data, self.variances)
            inputs_hr, offsets = get_hr_imgs(images_full_hr, bbox, images_full_hr.shape[3], images_full_hr.shape[2], images_full_hr.device)
            inputs_hr = inputs_hr.view(-1, 3, res_hr//8, res_hr//8)
            inputs_hr = inputs_hr[inputs_hr.view(-1, 3*(res_hr//8)*(res_hr//8)).sum(dim=1) != 0]
            out_hr = self.enhance_body(inputs_hr)
            fc_in = list(out_hr.values())[2]
            fc_in = self.avgpool(fc_in).view(fc_in.size(0), -1)
            fc_out = self.fc(fc_in)
            fc_gaze = self.FCGazeHead(fc_out)
            fc_hdps = self.FCHdpsHead(fc_out)
            fc_landmark = self.FCLandmarkHead(fc_out)

        features = self._fuse_with_attention(features, out_hr, offsets)

        if self.phase == 'train':
            bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
            classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
        gaze_regressions = torch.cat([self.GazeHead[i](feature) for i, feature in enumerate(features)], dim=1)
        hdps_regressions = torch.cat([self.HdpsHead[i](feature) for i, feature in enumerate(features)], dim=1)
        distance_regressions = torch.cat([self.DistHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions, gaze_regressions, 
                      hdps_regressions, distance_regressions, fc_gaze, fc_hdps, fc_landmark, bbox_regressions_pre, classifications_pre)
        else:
            output = (bbox_regressions_pre, F.softmax(classifications_pre, dim=-1), ldm_regressions, gaze_regressions, 
                      hdps_regressions, distance_regressions, fc_gaze, fc_hdps, fc_landmark)
        return output
