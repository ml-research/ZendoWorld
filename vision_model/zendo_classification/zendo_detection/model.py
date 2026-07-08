import torch 
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ZendoBackbone(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.conv_proj = nn.Conv2d(512, output_dim, kernel_size=1)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.conv_proj(feat)
        feat = feat.flatten(2).permute(0, 2, 1)
        return feat  # [B, N, C]

class ZendoImageToVectorModel(nn.Module):
    def __init__(self, config, num_output_tokens=7, token_dim=256,
                 num_colors=4, num_shapes=4, num_orientations=5, max_objects=7, dropout=0.1):
        super().__init__()
        self.token_dim = token_dim
        self.num_output_tokens = num_output_tokens
        self.max_objects = max_objects
        self.image_width = 640
        self.image_height = 480

        self.backbone = ZendoBackbone(output_dim=token_dim)
        self.query_tokens = nn.Parameter(torch.randn(1, num_output_tokens, token_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=8, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config["layers"])

        # Regularization
        self.norm = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim))

        # Heads
        self.color_head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(token_dim, num_colors)
        ) if config["color_mult_layer"] else nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(token_dim, num_colors)
        )
        
        self.shape_head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(token_dim, num_shapes)
        ) if config["shape_mult_layer"] else nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(token_dim, num_shapes)
        )

        self.orientation_head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(token_dim, num_orientations)
        ) if config["orientation_mult_layer"] else nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(token_dim, num_orientations)
        )

        self.presence_head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(token_dim, 1)
        ) if config["presence_mult_layer"] else nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(token_dim, 1)
        )
    
        self.pointing_head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(token_dim, max_objects + 2)
        ) if config["pointing_mult_layer"] else nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(token_dim, max_objects + 2)
        )

        self.touching_head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(token_dim, (max_objects + 2) * 6)
        ) if config["touching_mult_layer"] else nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(token_dim, (max_objects + 2) * 6)
        )

        self.bbox_head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(token_dim, 4),
            nn.Sigmoid()
        ) if config["bbox_mult_layer"] else nn.Sequential(
            nn.Linear(token_dim, 4),
            nn.Sigmoid()
        )

    def forward(self, images):
        features = self.backbone(images)

        queries = self.query_tokens.expand(images.size(0), -1, -1)
        cls = self.cls_token.expand(images.size(0), 1, -1)
        x = torch.cat([cls, features, queries], dim=1)
        x = self.transformer(x)

        output_tokens = x[:, -self.num_output_tokens:, :]
        scene_summary = x[:, 0, :]

        # Apply normalization and dropout
        output_tokens = self.norm(output_tokens)
        scene_summary = self.norm(scene_summary)
        output_tokens = self.dropout(output_tokens)

        # Heads
        touching_raw = self.touching_head(output_tokens)
        touching_logits = touching_raw.view(images.size(0), self.num_output_tokens, 6, self.max_objects + 2)
        # Scale bbox to match pixel dimensions
        bbox_raw = self.bbox_head(output_tokens)
        bbox_scaled = bbox_raw.clone()
        bbox_scaled[..., 0] *= self.image_width   # x_min
        bbox_scaled[..., 1] *= self.image_width  # x_max
        bbox_scaled[..., 2] *= self.image_height   # y_min
        bbox_scaled[..., 3] *= self.image_height  # y_max

        return {          # [B, T]
            "color": self.color_head(output_tokens),             # [B, T, num_colors]
            "shape": self.shape_head(output_tokens),             # [B, T, num_shapes]
            "orientation": self.orientation_head(output_tokens), # [B, T, num_orientations]
            "bbox": bbox_scaled,
            "pointing": self.pointing_head(output_tokens),
            "touching": touching_logits,
            "presence": self.presence_head(output_tokens),       # [B, T, 1]
        }

class ZendoSimpleModel(nn.Module):
    def __init__(self, config, num_output_tokens=7, token_dim=256,
                 num_colors=4, num_shapes=4, num_orientations=5, max_objects=7, dropout=0.1):
        super().__init__()
        self.token_dim = token_dim
        self.num_output_tokens = num_output_tokens
        self.max_objects = max_objects
        self.image_width = 640
        self.image_height = 480

        # CNN backbone
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-2])  # [B, 512, H/32, W/32]

        # Reduce to [B, num_output_tokens, token_dim]
        self.project = nn.Sequential(
            nn.AdaptiveAvgPool2d((num_output_tokens, 1)),  # force N "slots" from spatial map
            nn.Flatten(2),  # [B, C, N]
            nn.Conv1d(512, token_dim, kernel_size=1),  # [B, token_dim, N]
            nn.Dropout(p=dropout)
        )

        # Heads — optional extra MLP if config says so
        def make_head(out_dim, use_mult):
            return nn.Sequential(
                nn.Linear(token_dim, token_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(token_dim, out_dim)
            ) if use_mult else nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(token_dim, out_dim)
            )

        self.color_head = make_head(num_colors, config.get("color_mult_layer", False))
        self.shape_head = make_head(num_shapes, config.get("shape_mult_layer", False))
        self.orientation_head = make_head(num_orientations, config.get("orientation_mult_layer", False))
        self.presence_head = make_head(1, config.get("presence_mult_layer", False))
        self.pointing_head = make_head(max_objects + 2, config.get("pointing_mult_layer", False))
        self.touching_head = make_head((max_objects + 2) * 6, config.get("touching_mult_layer", False))
        self.bbox_head = nn.Sequential(
            nn.Linear(token_dim, 4),
            nn.Sigmoid()
        ) if not config.get("bbox_mult_layer", False) else nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(token_dim, 4),
            nn.Sigmoid()
        )

    def forward(self, images):
        x = self.backbone(images)  # [B, 512, H/32, W/32]
        x = self.project(x)        # [B, token_dim, num_output_tokens]
        x = x.permute(0, 2, 1)     # [B, num_output_tokens, token_dim]

        # Heads
        touching_raw = self.touching_head(x)
        touching_logits = touching_raw.view(images.size(0), self.num_output_tokens, 6, self.max_objects + 2)

        bbox_raw = self.bbox_head(x)
        bbox_scaled = bbox_raw.clone()
        bbox_scaled[..., 0] *= self.image_width
        bbox_scaled[..., 1] *= self.image_width
        bbox_scaled[..., 2] *= self.image_height
        bbox_scaled[..., 3] *= self.image_height

        return {
            "color": self.color_head(x),
            "shape": self.shape_head(x),
            "orientation": self.orientation_head(x),
            "bbox": bbox_scaled,
            "pointing": self.pointing_head(x),
            "touching": touching_logits,
            "presence": self.presence_head(x),
        }
    

class SmallCNNBackbone(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=128, out_dim=256, num_tokens=7):
        super().__init__()
        self.num_tokens = num_tokens
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),  # 320×240
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),

            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),  # 160×120
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),

            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),  # 80×60
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),

            nn.Conv2d(hidden_dim * 4, out_dim, kernel_size=3, stride=2, padding=1),  # 40×30
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((num_tokens, 1))  # [B, C, num_tokens, 1]

    def forward(self, x):
        x = self.encoder(x)  # [B, C, H, W]
        x = self.pool(x)     # [B, C, T, 1]
        x = x.squeeze(-1).permute(0, 2, 1)  # [B, T, C]
        return x


class ZendoLightweightModel(nn.Module):
    def __init__(self, config, num_output_tokens=7, token_dim=256,
                 num_colors=4, num_shapes=4, num_orientations=5, max_objects=7, dropout=0.1):
        super().__init__()
        self.token_dim = token_dim
        self.num_output_tokens = num_output_tokens
        self.max_objects = max_objects
        self.image_width = 640
        self.image_height = 480

        # Lightweight custom CNN backbone
        self.backbone = SmallCNNBackbone(out_dim=token_dim, num_tokens=num_output_tokens)

        # Prediction heads
        def make_head(out_dim, use_mult):
            return nn.Sequential(
                nn.Linear(token_dim, token_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(token_dim, out_dim)
            ) if use_mult else nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(token_dim, out_dim)
            )

        self.color_head = make_head(num_colors, config.get("color_mult_layer", False))
        self.shape_head = make_head(num_shapes, config.get("shape_mult_layer", False))
        self.orientation_head = make_head(num_orientations, config.get("orientation_mult_layer", False))
        self.presence_head = make_head(1, config.get("presence_mult_layer", False))
        self.pointing_head = make_head(max_objects + 2, config.get("pointing_mult_layer", False))
        self.touching_head = make_head((max_objects + 2) * 6, config.get("touching_mult_layer", False))
        self.bbox_head = nn.Sequential(
            nn.Linear(token_dim, 4),
            nn.Sigmoid()
        ) if not config.get("bbox_mult_layer", False) else nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(token_dim, 4),
            nn.Sigmoid()
        )

    def forward(self, images):
        x = self.backbone(images)  # [B, T, C]

        touching_raw = self.touching_head(x)
        touching_logits = touching_raw.view(images.size(0), self.num_output_tokens, 6, self.max_objects + 2)

        bbox_raw = self.bbox_head(x)
        bbox_scaled = bbox_raw.clone()
        bbox_scaled[..., 0] *= self.image_width
        bbox_scaled[..., 1] *= self.image_width
        bbox_scaled[..., 2] *= self.image_height
        bbox_scaled[..., 3] *= self.image_height

        return {
            "color": self.color_head(x),
            "shape": self.shape_head(x),
            "orientation": self.orientation_head(x),
            "bbox": bbox_scaled,
            "pointing": self.pointing_head(x),
            "touching": touching_logits,
            "presence": self.presence_head(x),
        }