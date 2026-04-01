"""
OmniVLA-edge model architecture.
Ported from https://github.com/NHirose/OmniVLA/blob/main/inference/model_omnivla_edge.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from efficientnet_pytorch import EfficientNet


# ── Positional encoding ───────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6):
        super().__init__()
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pos_enc', pos_enc.unsqueeze(0))

    def forward(self, x):
        return x + self.pos_enc[:, :x.size(1), :]


# ── Transformer decoder with modality masking ─────────────────────────────────

class MultiLayerDecoder_mask3(nn.Module):
    def __init__(self, embed_dim=512, seq_len=6, output_layers=[256, 128, 64],
                 nhead=8, num_layers=8, ff_dim_factor=4):
        super().__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=ff_dim_factor * embed_dim,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)
        self.output_layers = nn.ModuleList([nn.Linear(embed_dim + 1, embed_dim)])
        self.output_layers.append(nn.Linear(embed_dim, output_layers[0]))
        for i in range(len(output_layers) - 1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i + 1]))

    def forward(self, x, src_key_padding_mask, avg_pool_mask, no_goal_mask):
        x = self.positional_encoding(x)
        x = self.sa_decoder(x, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is not None:
            avg_mask = torch.index_select(avg_pool_mask, 0, no_goal_mask).unsqueeze(-1)
            x = x * avg_mask
        x = torch.mean(x, dim=1)
        x = x.reshape(x.shape[0], -1)
        if no_goal_mask.sum().item() == 9:
            no_goal_mask = torch.tensor([9]).to(no_goal_mask.device)
        x = torch.cat((x, no_goal_mask.unsqueeze(1)), dim=1)
        for layer in self.output_layers:
            x = F.relu(layer(x))
        return x


# ── FiLM network (language conditioning) ─────────────────────────────────────

def _conv_bn_relu(in_ch, out_ch, k, s, p):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_ch),
    )

class _InitialFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            _conv_bn_relu(3, 128, 5, 2, 2),
            _conv_bn_relu(128, 128, 3, 2, 1),
            _conv_bn_relu(128, 128, 3, 2, 1),
        )
    def forward(self, x): return self.layers(x)

class _IntermediateFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            _conv_bn_relu(128, 256, 3, 2, 1),
            _conv_bn_relu(256, 512, 3, 2, 1),
            _conv_bn_relu(512, 1024, 3, 2, 1),
            _conv_bn_relu(1024, 1024, 3, 2, 1),
        )
    def forward(self, x): return self.layers(x)

class _FiLMTransform(nn.Module):
    def forward(self, x, gamma, beta):
        return gamma.view(x.size(0), x.size(1), 1, 1) * x + beta.view(x.size(0), x.size(1), 1, 1)

class _ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.film  = _FiLMTransform()
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, beta, gamma):
        x = self.relu1(self.conv1(x))
        identity = x
        x = self.relu2(self.film(self.norm2(self.conv2(x)), beta, gamma))
        return x + identity

class _FinalClassifier(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 512, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.global_pool(x).view(x.size(0), -1)
        return self.fc_layers(x)


class FiLMNetwork(nn.Module):
    """
    Processes a 224×224 RGB image conditioned on a language embedding.
    Input:  img [B, 3, 224, 224],  question [B, question_dim]
    Output: features [B, 1024, 2, 2]  (flatten → 4096-dim)
    """
    def __init__(self, num_res_blocks, num_classes, num_channels, question_dim):
        super().__init__()
        self.film_param_generator = nn.Linear(question_dim, 2 * num_res_blocks * num_channels)
        self.initial_feature_extractor = _InitialFeatureExtractor()
        self.residual_blocks = nn.ModuleList(
            [_ResidualBlock(num_channels + 2, num_channels) for _ in range(num_res_blocks)]
        )
        self.intermediate_feature_extractor = _IntermediateFeatureExtractor()
        self.final_classifier = _FinalClassifier(num_channels, num_classes)  # kept for weight compat
        self.num_res_blocks = num_res_blocks
        self.num_channels   = num_channels

    def forward(self, x, question):
        B = x.size(0)
        x = self.initial_feature_extractor(x)
        film_params = self.film_param_generator(question).view(B, self.num_res_blocks, 2, self.num_channels)
        d = x.size(2)
        coords = torch.linspace(-1, 1, d, device=x.device)
        coord_x = coords.expand(B, 1, d, d)
        coord_y = coords.view(d, 1).expand(B, 1, d, d)
        for i, block in enumerate(self.residual_blocks):
            x = torch.cat([x, coord_x, coord_y], dim=1)
            x = block(x, film_params[:, i, 0], film_params[:, i, 1])
        return self.intermediate_feature_extractor(x)


# ── Main model ────────────────────────────────────────────────────────────────

class OmniVLA_edge(nn.Module):
    """
    Lightweight OmniVLA model for robot navigation.

    Accepts up to 4 goal modalities (pose, satellite map, goal image, language)
    selected via modality_id at runtime.

    Action output: [B, len_traj_pred, 4] — cumulative waypoints (dx, dy, cosθ, sinθ)
                   in robot frame, normalized by metric_waypoint_spacing (0.1 m).
    """
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: int = 8,
        learn_angle: bool = True,
        obs_encoder: str = "efficientnet-b0",
        obs_encoding_size: int = 1024,
        late_fusion: bool = False,
        mha_num_attention_heads: int = 4,
        mha_num_attention_layers: int = 4,
        mha_ff_dim_factor: int = 4,
    ):
        super().__init__()
        self.context_size       = context_size
        self.learn_angle        = learn_angle
        self.len_trajectory_pred = len_traj_pred
        self.num_action_params  = 4 if learn_angle else 2
        self.obs_encoding_size  = obs_encoding_size
        self.late_fusion        = late_fusion

        # ── Encoders ──────────────────────────────────────────────────────────
        assert obs_encoder.startswith("efficientnet"), "Only EfficientNet obs encoders supported"

        self.obs_encoder      = EfficientNet.from_name(obs_encoder, in_channels=3)
        self.goal_encoder     = EfficientNet.from_name("efficientnet-b0", in_channels=9)
        self.goal_encoder_img = EfficientNet.from_name(
            "efficientnet-b0", in_channels=3 if late_fusion else 6
        )

        obs_feat_dim     = self.obs_encoder._fc.in_features       # 1280 for B0
        map_feat_dim     = self.goal_encoder._fc.in_features       # 1280 for B0
        goal_img_feat_dim = self.goal_encoder_img._fc.in_features  # 1280 for B0
        lang_feat_dim    = 4096   # FiLMNetwork output flattened

        self.compress_obs_enc      = nn.Linear(obs_feat_dim,      obs_encoding_size)
        self.compress_obs_enc_map  = nn.Linear(map_feat_dim,      obs_encoding_size)
        self.compress_goal_enc_img = nn.Linear(goal_img_feat_dim, obs_encoding_size)
        self.compress_goal_enc_lan = nn.Linear(lang_feat_dim,     obs_encoding_size)

        # Pose encoder: 4D → encoding_size
        self.local_goal = nn.Sequential(nn.Linear(4, obs_encoding_size))

        # FiLM for language conditioning (question_dim=512 matches CLIP ViT-B/32)
        self.film_model = FiLMNetwork(
            num_res_blocks=8, num_classes=10, num_channels=128, question_dim=512
        )

        # ── Transformer decoder ───────────────────────────────────────────────
        # seq_len = context_size+1 obs frames + pose + map + goal_img + language
        seq_len = context_size + 1 + 4
        self.decoder = MultiLayerDecoder_mask3(
            embed_dim=obs_encoding_size,
            seq_len=seq_len,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )

        # ── Action / distance heads ───────────────────────────────────────────
        self.action_predictor = nn.Sequential(nn.Linear(32, len_traj_pred * self.num_action_params))
        self.dist_predictor   = nn.Sequential(nn.Linear(32, 1))

        # ── Modality attention masks ──────────────────────────────────────────
        # seq positions: [0..ctx] = obs, [-4]=pose, [-3]=map, [-2]=goal_img, [-1]=language
        T = seq_len

        def make_mask(*masked_positions):
            m = torch.zeros(1, T, dtype=torch.bool)
            for p in masked_positions:
                m[:, p] = True
            return m

        # goal_mask_X: True = token is MASKED OUT (ignored by attention)
        self.goal_mask_0 = make_mask(-4, -2, -1)   # satellite only
        self.goal_mask_1 = make_mask(-3, -2, -1)   # pose only (wait: mask map,img,lan → pose+obs visible)
        self.goal_mask_2 = make_mask(-4, -2, -1)   # satellite only (same as 0, kept for index compat)
        self.goal_mask_3 = make_mask(-4, -1)        # satellite + image
        self.goal_mask_4 = make_mask(-3, -1)        # pose + satellite
        self.goal_mask_5 = make_mask(-1)            # all except language
        self.goal_mask_6 = make_mask(-4, -3, -1)   # image only
        self.goal_mask_7 = make_mask(-4, -3, -2)   # language only
        self.goal_mask_8 = make_mask(-3, -2)        # language + pose

        # all_masks order matches modality_id mapping in inference script
        self.all_masks = torch.cat([
            self.goal_mask_0,  # id=0: satellite only
            self.goal_mask_2,  # id=1: satellite+image (re-uses mask_2)
            self.goal_mask_3,  # id=2: satellite+image
            self.goal_mask_5,  # id=3: all
            self.goal_mask_1,  # id=4: pose only
            self.goal_mask_4,  # id=5: pose+image
            self.goal_mask_6,  # id=6: image only
            self.goal_mask_7,  # id=7: language only
            self.goal_mask_8,  # id=8: language+pose
        ], dim=0)

        # Weighted avg-pool mask: up-weight active tokens so mean pool is unbiased
        def avep(mask):
            active = (1.0 - mask.float())
            return active * (T / active.sum())
        self.avg_pool_mask = torch.cat([avep(m) for m in [
            self.goal_mask_0, self.goal_mask_2, self.goal_mask_3, self.goal_mask_5,
            self.goal_mask_1, self.goal_mask_4, self.goal_mask_6, self.goal_mask_7,
            self.goal_mask_8,
        ]], dim=0)

    # ── EfficientNet feature extraction helper ─────────────────────────────────

    @staticmethod
    def _eff_encode(net, x):
        z = net.extract_features(x)
        z = net._avg_pooling(z)
        if net._global_params.include_top:
            z = z.flatten(start_dim=1)
            z = net._dropout(z)
        return z

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        obs_img:     torch.Tensor,   # [B, 3*(ctx+1), 96, 96]
        goal_pose:   torch.Tensor,   # [B, 4]
        map_images:  torch.Tensor,   # [B, 9, 352, 352]
        goal_img:    torch.Tensor,   # [B, 3, 96, 96]
        goal_mask:   torch.Tensor,   # [B]  modality_id per sample
        feat_text:   torch.Tensor,   # [B, 512]  CLIP text features
        current_img: torch.Tensor,   # [B, 3, 224, 224]  for FiLM
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = obs_img.device
        B = obs_img.size(0)

        # ── Language token (FiLM + CLIP) ──────────────────────────────────────
        film_out = self.film_model(current_img, feat_text)           # [B, 1024, 2, 2]
        lang_enc = self.compress_goal_enc_lan(film_out.flatten(1))   # [B, enc_size]
        lang_enc = lang_enc.unsqueeze(1)                              # [B, 1, enc_size]

        # ── Goal image token ──────────────────────────────────────────────────
        cur_frame = obs_img[:, 3 * self.context_size:, :, :]         # last obs frame [B,3,96,96]
        if self.late_fusion:
            goal_enc_img = self._eff_encode(self.goal_encoder_img, goal_img)
        else:
            obsgoal = torch.cat([cur_frame, goal_img], dim=1)        # [B, 6, 96, 96]
            goal_enc_img = self._eff_encode(self.goal_encoder_img, obsgoal)
        goal_enc_img = self.compress_goal_enc_img(goal_enc_img).unsqueeze(1)  # [B,1,enc]

        # ── Pose token ────────────────────────────────────────────────────────
        pose_enc = self.local_goal(goal_pose).unsqueeze(1)           # [B, 1, enc_size]

        # ── Satellite map token ───────────────────────────────────────────────
        map_enc = self._eff_encode(self.goal_encoder, map_images)
        map_enc = self.compress_obs_enc_map(map_enc).unsqueeze(1)    # [B, 1, enc_size]

        # ── Observation tokens (context_size+1 frames) ────────────────────────
        frames = torch.split(obs_img, 3, dim=1)                      # tuple of [B,3,96,96]
        frames = torch.cat(frames, dim=0)                            # [B*(ctx+1), 3, 96, 96]
        obs_enc = self._eff_encode(self.obs_encoder, frames)
        obs_enc = self.compress_obs_enc(obs_enc)                     # [B*(ctx+1), enc_size]
        obs_enc = obs_enc.reshape(self.context_size + 1, B, self.obs_encoding_size)
        obs_enc = obs_enc.permute(1, 0, 2)                           # [B, ctx+1, enc_size]

        # ── Concatenate all tokens ────────────────────────────────────────────
        # Order: [obs×(ctx+1), pose, map, goal_img, language]
        tokens = torch.cat([obs_enc, pose_enc, map_enc, goal_enc_img, lang_enc], dim=1)

        # ── Transformer with modality masking ─────────────────────────────────
        no_goal_mask      = goal_mask.long()
        pad_mask          = torch.index_select(self.all_masks.to(device), 0, no_goal_mask)
        avg_mask          = self.avg_pool_mask.to(device)

        feat = self.decoder(tokens, pad_mask, avg_mask, no_goal_mask)

        # ── Action prediction ─────────────────────────────────────────────────
        action_pred = self.action_predictor(feat)
        action_pred = action_pred.reshape(B, self.len_trajectory_pred, self.num_action_params)
        action_pred[:, :, :2] = torch.cumsum(action_pred[:, :, :2], dim=1)  # delta → absolute
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(action_pred[:, :, 2:].clone(), dim=-1)

        dist_pred = self.dist_predictor(feat)
        return action_pred, dist_pred, no_goal_mask
