"""Fast checks for the core model and HRAL implementation."""

import torch

from spikevimformer import build_model
from spikevimformer.losses import hral_alignment_loss, rerank_features


def test_tiny_model_train_and_eval_outputs():
    torch.set_num_threads(1)
    model = build_model(num_classes=3, variant="tiny")
    images = torch.randn(2, 3, 32, 32)

    model.eval()
    with torch.inference_mode():
        inference = model(images)
    assert inference.descriptor.shape == (2, 240)
    assert inference.ssa_descriptor is None

    model.train()
    query, gallery = model(images, images.flip(-1))
    assert query.ssa_descriptor.shape == (2, 240)
    assert gallery.shs_logits.shape == (2, 3)


def test_hral_is_finite_and_shape_preserving():
    query = torch.randn(4, 16, requires_grad=True)
    gallery = torch.randn(4, 16, requires_grad=True)
    refined_query, refined_gallery = rerank_features(query, gallery, top_k=3)
    assert refined_query.shape == query.shape
    assert refined_gallery.shape == gallery.shape
    loss = hral_alignment_loss(query, gallery, top_k=3)
    assert torch.isfinite(loss)
    loss.backward()
    assert query.grad is not None


def test_official_backbone_checkpoint_loading(tmp_path):
    source = build_model(num_classes=3, variant="tiny")
    checkpoint = {
        "model": {
            **source.backbone.state_dict(),
            "head.weight": torch.randn(1000, 240),
            "head.bias": torch.randn(1000),
        }
    }
    checkpoint_path = tmp_path / "V3_10.0M_1x4.pth"
    torch.save(checkpoint, checkpoint_path)

    target = build_model(num_classes=3, variant="tiny")
    target.load_backbone_checkpoint(str(checkpoint_path))
    for key, value in source.backbone.state_dict().items():
        assert torch.equal(value, target.backbone.state_dict()[key])
