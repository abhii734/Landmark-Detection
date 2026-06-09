"""
Landmark Detection - Model Architectures
CNN models for landmark classification and retrieval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm


class LandmarkClassifier(nn.Module):
    """
    Landmark Classifier using pretrained backbone.
    
    Args:
        num_classes: Number of landmark classes
        backbone: Backbone architecture (resnet50, efficientnet_b0, etc.)
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
    """
    
    def __init__(self, num_classes, backbone='resnet50', pretrained=True, dropout=0.3):
        super(LandmarkClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Create backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone.startswith('vit_'):
            # Vision Transformer from timm
            self.backbone = timm.create_model(backbone, pretrained=pretrained)
            feature_dim = self.backbone.num_features
            self.head = nn.Identity()
        else:
            # Use timm models
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            feature_dim = self.backbone.num_features
        
        self.feature_dim = feature_dim
        
        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
    def forward(self, x):
        """Forward pass."""
        features = self.backbone(x)
        logits = self.head(features)
        return logits
    
    def get_features(self, x):
        """Get embedding features without classification."""
        features = self.backbone(x)
        return features


class LandmarkEmbeddingModel(nn.Module):
    """
    Landmark model with embedding output for retrieval tasks.
    Uses ArcFace loss for better embedding learning.
    
    Args:
        num_classes: Number of landmark classes
        embedding_dim: Dimension of embedding space
        backbone: Backbone architecture
        pretrained: Use pretrained weights
    """
    
    def __init__(self, num_classes, embedding_dim=512, backbone='resnet50', pretrained=True):
        super(LandmarkEmbeddingModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # Create backbone
        if backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            feature_dim = base_model.fc.in_features
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        elif backbone == 'efficientnet_b0':
            base_model = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = base_model.classifier[1].in_features
            self.backbone = nn.Sequential(
                base_model.features,
                base_model.avgpool
            )
        else:
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            feature_dim = self.backbone.num_features
            self.backbone = nn.Identity()
            feature_dim = self.backbone.num_features if hasattr(self.backbone, 'num_features') else 2048
        
        # Embedding layer with BatchNorm
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # Classification head for ArcFace
        self.arcface_head = ArcFaceHead(embedding_dim, num_classes)
    
    def forward(self, x, labels=None, mode='train'):
        """
        Forward pass.
        
        Args:
            x: Input images [B, C, H, W]
            labels: Class labels for ArcFace training
            mode: 'train' for ArcFace, 'eval' for normal features
            
        Returns:
            If mode='train' and labels provided: (logits, embeddings)
            Otherwise: embeddings
        """
        # Extract features
        if isinstance(self.backbone, nn.Sequential):
            features = self.backbone(x)
            features = features.view(features.size(0), -1)
        else:
            features = self.backbone(x)
        
        # Get embeddings
        embeddings = self.embedding(features)
        
        if mode == 'train' and labels is not None:
            logits = self.arcface_head(embeddings, labels)
            return logits, embeddings
        else:
            # Normalize embeddings for retrieval
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings


class ArcFaceHead(nn.Module):
    """
    ArcFace head for metric learning.
    
    Paper: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    """
    
    def __init__(self, embedding_dim, num_classes, s=30.0, m=0.5):
        super(ArcFaceHead, self).__init__()
        self.s = s  # Scale
        self.m = m  # Angular margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(m) if 'math' in dir() else torch.cos(torch.tensor(m))
        self.sin_m = math.sin(m) if 'math' in dir() else torch.sin(torch.tensor(m))
        self.th = torch.cos(math.pi - m) if 'math' in dir() else torch.cos(torch.tensor(torch.pi - m))
        self.mm = torch.sin(math.pi - m) * m if 'math' in dir() else torch.sin(torch.tensor(torch.pi - m)) * m
    
    def forward(self, embeddings, labels):
        # Normalize weights and embeddings
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        
        # Calculate adjusted cosine with angular margin
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output


class LandmarkModelWithAttention(nn.Module):
    """
    Landmark model with attention mechanism for better feature extraction.
    """
    
    def __init__(self, num_classes, embedding_dim=512, backbone='resnet50', pretrained=True):
        super(LandmarkModelWithAttention, self).__init__()
        
        # Backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Identity()
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        
        # Embedding
        self.embedding = nn.Sequential(
            nn.Linear(2048, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # Classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        
        # Attention weights (using first feature as query)
        attn_weights = self.attention(features)
        
        # Weighted features
        weighted_features = features * attn_weights
        
        # Embeddings
        embeddings = self.embedding(weighted_features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Classification
        logits = self.classifier(embeddings)
        
        return logits, embeddings


class GeM(nn.Module):
    """
    Generalized Mean Pooling - better than average pooling for retrieval.
    """
    
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    
    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1./self.p)


def get_model(model_name, num_classes, pretrained=True, **kwargs):
    """
    Factory function to get landmark model.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of classes
        pretrained: Use pretrained weights
        
    Returns:
        Model instance
    """
    models = {
        'resnet50': LandmarkClassifier,
        'efficientnet_b0': LandmarkClassifier,
        'efficientnet_b3': LandmarkClassifier,
        'arcface_resnet50': LandmarkEmbeddingModel,
        'attention_resnet50': LandmarkModelWithAttention
    }
    
    if model_name in models:
        return models[model_name](num_classes=num_classes, pretrained=pretrained, **kwargs)
    else:
        # Use timm models
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        return model


if __name__ == "__main__":
    # Test models
    print("Testing LandmarkClassifier...")
    model = LandmarkClassifier(num_classes=1000, backbone='resnet50')
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")
    
    print("\nTesting LandmarkEmbeddingModel...")
    model = LandmarkEmbeddingModel(num_classes=1000, embedding_dim=512)
    embeddings = model(x)
    print(f"Embedding shape: {embeddings.shape}")