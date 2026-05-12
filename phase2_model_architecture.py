"""
CVPR 2026 - The First AI for Children Challenge
Phase 2: Model Architecture - ST-GCN for Gait Analysis

Author: AI Expert Team
Description: Spatial-Temporal Graph Convolutional Networks for children gait analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Note: torch_geometric is optional - we implement our own GCN layers


# ============================================================================
# SKELETON DEFINITION FOR HUMAN POSE
# ============================================================================

class SkeletonConfig:
    """Human skeleton configuration for graph construction"""
    
    # COCO-style 17 keypoints
    NUM_JOINTS = 17
    
    # Joint names (COCO format)
    JOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # Natural adjacency matrix (bone connections)
    EDGES = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6),  # Shoulders
        (5, 7), (7, 9),  # Left arm
        (6, 8), (8, 10),  # Right arm
        (5, 11), (6, 12),  # Shoulder to hip
        (11, 12),  # Hips
        (11, 13), (13, 15),  # Left leg
        (12, 14), (14, 16),  # Right leg
    ]
    
    @staticmethod
    def get_adjacency_matrix():
        """Create adjacency matrix from edges"""
        adj = np.zeros((SkeletonConfig.NUM_JOINTS, SkeletonConfig.NUM_JOINTS))
        for i, j in SkeletonConfig.EDGES:
            adj[i, j] = 1
            adj[j, i] = 1
        # Self-loops
        np.fill_diagonal(adj, 1)
        return adj
    
    @staticmethod
    def normalize_adjacency(adj):
        """Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)"""
        degree = np.sum(adj, axis=1)
        degree_inv_sqrt = np.power(degree + 1e-8, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.
        degree_mat_inv_sqrt = np.diag(degree_inv_sqrt)
        normalized_adj = adj.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt)
        return normalized_adj


# ============================================================================
# GRAPH CONVOLUTION LAYER
# ============================================================================

class GraphConvolution(nn.Module):
    """
    Graph Convolution Layer for spatial feature extraction
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Args:
            x: (batch, num_joints, channels)
            adj: (num_joints, num_joints) - adjacency matrix
        Returns:
            (batch, num_joints, out_channels)
        """
        # x: [B, N, C], adj: [N, N]
        # Graph convolution: A * X * W
        support = torch.matmul(x, self.weight)  # [B, N, C_out]
        output = torch.matmul(adj, support)  # [B, N, C_out]
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


# ============================================================================
# ST-GCN BLOCK
# ============================================================================

class STGCNBlock(nn.Module):
    """
    Spatial-Temporal Graph Convolution Block
    Combines spatial GCN with temporal convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, 
                 residual=True, dropout=0.1):
        super(STGCNBlock, self).__init__()
        
        # Spatial GCN
        self.gcn = GraphConvolution(in_channels, out_channels)
        self.bn_spatial = nn.BatchNorm1d(SkeletonConfig.NUM_JOINTS)
        
        # Temporal Convolution
        padding = (kernel_size - 1) // 2
        self.tcn = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=out_channels  # Depthwise convolution
        )
        self.bn_temporal = nn.BatchNorm1d(out_channels)
        
        # Residual connection
        if residual:
            if stride != 1 or in_channels != out_channels:
                self.residual = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm1d(out_channels)
                )
            else:
                self.residual = nn.Identity()
        else:
            self.residual = None
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, adj):
        """
        Args:
            x: (batch, seq_len, num_joints, channels)
            adj: (num_joints, num_joints)
        Returns:
            (batch, seq_len, num_joints, out_channels)
        """
        B, T, N, C = x.shape
        
        # Reshape for spatial GCN: [B*T, N, C]
        x_spatial = x.reshape(B * T, N, C)  # Use reshape instead of view
        
        # Spatial GCN
        x_spatial = self.gcn(x_spatial, adj)  # [B*T, N, C_out]
        x_spatial = self.bn_spatial(x_spatial)
        x_spatial = self.relu(x_spatial)
        
        # Reshape back: [B, T, N, C_out] -> [B, C_out, T, N]
        x_spatial = x_spatial.reshape(B, T, N, -1)  # Use reshape
        x_spatial = x_spatial.permute(0, 3, 1, 2).contiguous()  # [B, C_out, T, N]
        
        # Apply TCN on each joint independently
        x_tcn = []
        for i in range(N):
            joint_feat = x_spatial[:, :, :, i]  # [B, C_out, T]
            joint_feat = self.tcn(joint_feat)  # [B, C_out, T']
            joint_feat = self.bn_temporal(joint_feat)
            joint_feat = self.relu(joint_feat)
            x_tcn.append(joint_feat.unsqueeze(-1))
        
        x_tcn = torch.cat(x_tcn, dim=-1)  # [B, C_out, T', N]
        x_tcn = x_tcn.permute(0, 2, 3, 1).contiguous()  # [B, T', N, C_out]
        
        # Handle stride for residual - reshape to [B*N, C, T] for Conv1d
        if self.residual is not None:
            # Reshape x from [B, T, N, C] to [B*N, C, T] for Conv1d
            x_for_conv = x.permute(0, 2, 3, 1).contiguous()  # [B, N, C, T]
            x_for_conv = x_for_conv.reshape(B * N, C, T)  # [B*N, C, T]
            x_res = self.residual(x_for_conv)  # [B*N, C_out, T']
            
            # Get new T' after stride
            T_prime = x_res.shape[-1]
            
            # Reshape back to [B, T', N, C_out]
            x_res = x_res.reshape(B, N, -1, T_prime)  # [B, N, C_out, T']
            x_res = x_res.permute(0, 3, 1, 2).contiguous()  # [B, T', N, C_out]
        else:
            x_res = x[:, :x_tcn.size(1), :, :]  # Truncate if needed
            T_prime = x_tcn.size(1)
        
        # Residual connection
        output = x_tcn + x_res
        output = self.dropout(output)
        output = self.relu(output)
        
        return output


# ============================================================================
# ST-GCN BACKBONE
# ============================================================================

class STGCNBackbone(nn.Module):
    """
    Complete ST-GCN backbone for gait analysis
    Extracts spatio-temporal features from keypoint sequences
    """
    def __init__(self, in_channels=2, base_channels=64, num_blocks=4, dropout=0.1):
        super(STGCNBackbone, self).__init__()
        
        self.num_blocks = num_blocks
        self.base_channels = base_channels
        
        # Get normalized adjacency matrix
        adj = SkeletonConfig.get_adjacency_matrix()
        adj_norm = SkeletonConfig.normalize_adjacency(adj)
        self.register_buffer('adjacency', torch.FloatTensor(adj_norm))
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, base_channels)
        self.bn_input = nn.BatchNorm1d(base_channels)  # BN on channels
        
        # ST-GCN blocks
        channels = [base_channels]
        for i in range(num_blocks):
            channels.append(base_channels * (2 ** i))
        
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = STGCNBlock(
                in_channels=channels[i],
                out_channels=channels[i+1],
                kernel_size=9 if i == 0 else 5,
                stride=2 if i > 0 else 1,
                residual=True,
                dropout=dropout
            )
            self.blocks.append(block)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, num_joints, channels)
        Returns:
            Global features: (batch, final_channels)
            Per-joint features: (batch, num_joints, final_channels)
        """
        B, T, N, C_in = x.shape
        
        # Input projection
        x = self.input_proj(x)  # [B, T, N, base_channels]
        _, _, _, C = x.shape  # Get new channel count after projection
        
        # Apply batch normalization on channels dimension
        # Reshape to [B*T*N, C] for BN
        x_flat = x.view(B * T * N, C)  # [B*T*N, C]
        x_bn = self.bn_input(x_flat)  # BN on channels
        x = x_bn.view(B, T, N, C)  # [B, T, N, C]
        x = self.relu(x)
        
        # ST-GCN blocks
        for block in self.blocks:
            x = block(x, self.adjacency)
        
        # x: [B, T', N, C_final]
        B, T_prime, N, C_final = x.shape
        
        # Global temporal pooling - average over time
        x_temporal = x.mean(dim=1)  # [B, N, C_final]
        
        # Global joint pooling - average over joints
        x_global = x_temporal.mean(dim=1)  # [B, C_final]
        
        # Per-joint features (for fine-grained analysis)
        x_joint = x_temporal  # [B, N, C_final]
        
        return x_global, x_joint


# ============================================================================
# TRACK 1 HEAD - EVGS SCORING
# ============================================================================

class Track1Head(nn.Module):
    """
    Multi-task head for Track 1:
    - 34 binary classifications (17 items × 2 limbs)
    - 2 regression outputs (total scores for left/right)
    """
    def __init__(self, feature_dim, num_items=17, hidden_dim=128):
        super(Track1Head, self).__init__()
        
        self.num_items = num_items
        self.total_labels = num_items * 2  # 34
        
        # Binary classification heads (one per item per limb)
        self.binary_heads = nn.ModuleList()
        for _ in range(self.total_labels):
            head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 1)
            )
            self.binary_heads.append(head)
        
        # Regression heads for total scores
        self.regression_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)  # Left total, Right total
        )
    
    def forward(self, features, joint_features=None):
        """
        Args:
            features: (batch, feature_dim) - global features
            joint_features: (batch, num_joints, feature_dim) - optional per-joint features
        Returns:
            binary_logits: (batch, 34)
            total_scores: (batch, 2)
        """
        # Binary predictions
        binary_outputs = []
        for head in self.binary_heads:
            out = head(features)  # [B, 1]
            binary_outputs.append(out.squeeze(-1))
        
        binary_logits = torch.stack(binary_outputs, dim=1)  # [B, 34]
        
        # Total score regression
        total_scores = self.regression_head(features)  # [B, 2]
        
        return binary_logits, total_scores


# ============================================================================
# TRACK 2 HEAD - GAIT CLASSIFICATION
# ============================================================================

class Track2Head(nn.Module):
    """
    Classification head for Track 2:
    - 5-class classification for left limb
    - 5-class classification for right limb
    """
    def __init__(self, feature_dim, num_classes=5, hidden_dim=256):
        super(Track2Head, self).__init__()
        
        self.num_classes = num_classes
        
        # Shared feature processing
        self.shared_fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )
        
        # Limb-specific heads
        self.left_head = nn.Linear(hidden_dim, num_classes)
        self.right_head = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, features, joint_features=None):
        """
        Args:
            features: (batch, feature_dim) - global features
            joint_features: (batch, num_joints, feature_dim) - optional per-joint features
        Returns:
            left_logits: (batch, num_classes)
            right_logits: (batch, num_classes)
        """
        # Shared processing
        shared = self.shared_fc(features)  # [B, hidden_dim]
        
        # Limb-specific predictions
        left_logits = self.left_head(shared)  # [B, num_classes]
        right_logits = self.right_head(shared)  # [B, num_classes]
        
        return left_logits, right_logits


# ============================================================================
# COMPLETE MODEL FOR BOTH TRACKS
# ============================================================================

class ChildrenGaitModel(nn.Module):
    """
    Unified model for both tracks:
    - Shared ST-GCN backbone
    - Track-specific heads
    """
    def __init__(self, in_channels=2, base_channels=64, num_blocks=4,
                 track1_num_items=17, track2_num_classes=5):
        super(ChildrenGaitModel, self).__init__()
        
        # Shared backbone
        self.backbone = STGCNBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            num_blocks=num_blocks
        )
        
        # Track 1 head
        feature_dim = base_channels * (2 ** (num_blocks - 1))
        self.track1_head = Track1Head(
            feature_dim=feature_dim,
            num_items=track1_num_items
        )
        
        # Track 2 head
        self.track2_head = Track2Head(
            feature_dim=feature_dim,
            num_classes=track2_num_classes
        )
    
    def forward(self, x, track='both'):
        """
        Args:
            x: (batch, seq_len, num_joints, channels)
            track: 'track1', 'track2', or 'both'
        Returns:
            Dictionary with track-specific outputs
        """
        # Extract features
        global_features, joint_features = self.backbone(x)
        
        outputs = {}
        
        if track in ['track1', 'both']:
            binary_logits, total_scores = self.track1_head(
                global_features, joint_features
            )
            outputs['track1_binary'] = binary_logits
            outputs['track1_total'] = total_scores
        
        if track in ['track2', 'both']:
            left_logits, right_logits = self.track2_head(
                global_features, joint_features
            )
            outputs['track2_left'] = left_logits
            outputs['track2_right'] = right_logits
        
        return outputs


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class Track1Loss(nn.Module):
    """
    Combined loss for Track 1:
    - BCE loss for binary classifications
    - MSE loss for total score regression
    """
    def __init__(self, bce_weight=1.0, mse_weight=0.5, class_weights=None):
        super(Track1Loss, self).__init__()
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight
        
        if class_weights is not None:
            self.class_weights = torch.FloatTensor(class_weights)
        else:
            self.class_weights = None
    
    def forward(self, binary_logits, binary_targets, total_pred, total_targets):
        """
        Args:
            binary_logits: (batch, 34)
            binary_targets: (batch, 34)
            total_pred: (batch, 2)
            total_targets: (batch, 2)
        """
        # Binary cross-entropy loss
        if self.class_weights is not None:
            # Weighted BCE
            pos_weights = self.class_weights.to(binary_logits.device)
            bce_loss = F.binary_cross_entropy_with_logits(
                binary_logits, binary_targets,
                pos_weight=pos_weights,
                reduction='mean'
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(
                binary_logits, binary_targets,
                reduction='mean'
            )
        
        # MSE loss for total scores
        mse_loss = F.mse_loss(total_pred, total_targets)
        
        total_loss = self.bce_weight * bce_loss + self.mse_weight * mse_loss
        
        return total_loss, {'bce': bce_loss.item(), 'mse': mse_loss.item()}


class Track2Loss(nn.Module):
    """
    Cross-entropy loss with class weights for Track 2
    """
    def __init__(self, class_weights=None, focal_gamma=2.0, use_focal=False):
        super(Track2Loss, self).__init__()
        
        if class_weights is not None:
            self.class_weights = torch.FloatTensor(class_weights)
        else:
            self.class_weights = None
        
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
    
    def forward(self, left_logits, left_targets, right_logits, right_targets):
        """
        Args:
            left_logits: (batch, num_classes)
            left_targets: (batch,)
            right_logits: (batch, num_classes)
            right_targets: (batch,)
        """
        # Combine left and right for batch processing
        all_logits = torch.cat([left_logits, right_logits], dim=0)  # [2B, C]
        all_targets = torch.cat([left_targets, right_targets], dim=0)  # [2B]
        
        if self.use_focal:
            # Focal Loss
            ce_loss = F.cross_entropy(
                all_logits, all_targets,
                weight=self.class_weights.to(all_logits.device) if self.class_weights is not None else None,
                reduction='none'
            )
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.focal_gamma * ce_loss).mean()
            total_loss = focal_loss
        else:
            # Standard CE with class weights
            total_loss = F.cross_entropy(
                all_logits, all_targets,
                weight=self.class_weights.to(all_logits.device) if self.class_weights is not None else None
            )
        
        return total_loss, {'ce': total_loss.item()}


# ============================================================================
# MODEL TESTING
# ============================================================================

def test_model():
    """Test the model architecture"""
    print("=" * 80)
    print("TESTING MODEL ARCHITECTURE")
    print("=" * 80)
    
    # Create dummy input
    batch_size = 4
    seq_len = 50
    num_joints = 17
    channels = 2
    
    x = torch.randn(batch_size, seq_len, num_joints, channels)
    print(f"Input shape: {x.shape}")
    
    # Create model
    model = ChildrenGaitModel(
        in_channels=channels,
        base_channels=64,
        num_blocks=4
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Forward pass - Track 1
    print("\n--- Track 1 Forward Pass ---")
    outputs_t1 = model(x, track='track1')
    print(f"Binary logits shape: {outputs_t1['track1_binary'].shape}")
    print(f"Total scores shape: {outputs_t1['track1_total'].shape}")
    
    # Forward pass - Track 2
    print("\n--- Track 2 Forward Pass ---")
    outputs_t2 = model(x, track='track2')
    print(f"Left logits shape: {outputs_t2['track2_left'].shape}")
    print(f"Right logits shape: {outputs_t2['track2_right'].shape}")
    
    # Forward pass - Both
    print("\n--- Both Tracks Forward Pass ---")
    outputs_both = model(x, track='both')
    print(f"All outputs: {list(outputs_both.keys())}")
    
    # Test loss functions
    print("\n--- Testing Loss Functions ---")
    
    # Track 1 loss
    binary_targets = torch.randint(0, 2, (batch_size, 34)).float()
    total_targets = torch.randn(batch_size, 2)
    
    loss_fn_t1 = Track1Loss()
    loss_t1, metrics_t1 = loss_fn_t1(
        outputs_both['track1_binary'], binary_targets,
        outputs_both['track1_total'], total_targets
    )
    print(f"Track 1 Loss: {loss_t1:.4f}, Metrics: {metrics_t1}")
    
    # Track 2 loss
    left_targets = torch.randint(0, 5, (batch_size,))
    right_targets = torch.randint(0, 5, (batch_size,))
    
    loss_fn_t2 = Track2Loss(class_weights=[0.8, 0.6, 0.6, 4.4, 4.4])
    loss_t2, metrics_t2 = loss_fn_t2(
        outputs_both['track2_left'], left_targets,
        outputs_both['track2_right'], right_targets
    )
    print(f"Track 2 Loss: {loss_t2:.4f}, Metrics: {metrics_t2}")
    
    print("\n✅ Model architecture test passed!")
    
    return model


if __name__ == "__main__":
    model = test_model()
