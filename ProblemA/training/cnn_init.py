"""核对公开RGB预训练权重的3→4通道迁移，不执行优化。"""
import torch


def verify_stem(model, checkpoint):
    source = torch.load(checkpoint, map_location='cpu', weights_only=True)
    rgb = source['stem.0.weight']
    actual = model.encoder.stem[0].weight.detach().cpu()
    torch.testing.assert_close(actual[:, :3], rgb, rtol=0, atol=0)
    torch.testing.assert_close(actual[:, 3:4], rgb.mean(1, keepdim=True), rtol=0, atol=0)
    assert sum(p.numel() for p in model.parameters()) == 64782788
    return dict(passed=True, rgb_weight_shape=list(rgb.shape), adapted_weight_shape=list(actual.shape),
        first_three_channels_equal_RGB=True, IR_channel_equals_RGB_weight_mean=True,
        parameters=64782788, optimizer_steps=0)
