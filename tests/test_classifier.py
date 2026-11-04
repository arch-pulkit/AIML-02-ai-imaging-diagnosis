import torch
from torchvision.models import densenet121, DenseNet121_Weights
from src.models.classifier import build_model
from src.data.nih import LABELS


def test_build_model_shape_and_params():
    model = build_model(num_classes=len(LABELS), pretrained=False)
    model.eval()
    
    # Check classifier head type and dimensions
    assert isinstance(model.classifier, torch.nn.Linear)
    assert model.classifier.out_features == len(LABELS)
    
    # Check output shape on dummy batch
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, len(LABELS))
    
    # Check parameter count for modified DenseNet121
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 6968206, f"Expected 6,968,206 params, got {total_params}"


def test_pretrained_weights():
    # Verify that pretrained=True loads ImageNet weights and pretrained=False initializes randomly
    model_pretrained = build_model(pretrained=True)
    model_random = build_model(pretrained=False)
    
    ref_weight = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).features.conv0.weight
    
    assert torch.equal(model_pretrained.features.conv0.weight, ref_weight)
    assert not torch.equal(model_random.features.conv0.weight, ref_weight)