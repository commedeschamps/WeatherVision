"""Equal-area deletion controls for a fixed class logit; exploratory, not causal proof."""
from __future__ import annotations
import operator
import numpy as np
import torch
from torchvision.transforms.functional import gaussian_blur


def region_selections(heatmap, grid=7, count=5, random_repeats=10, seed=42):
    cam = np.asarray(heatmap, dtype=np.float32)
    if cam.shape != (224, 224) or not np.isfinite(cam).all() or cam.min() < 0 or cam.max() > 1:
        raise ValueError('Expected a finite normalized 224-square heatmap')
    if not isinstance(grid, int) or grid < 2 or 224 % grid or not 1 <= count <= grid * grid // 2:
        raise ValueError('Choose an equal-size grid dividing 224 and a valid region budget')
    if not isinstance(random_repeats, int) or random_repeats < 1:
        raise ValueError('At least one random control is required')
    side = 224 // grid
    importance = cam.reshape(grid, side, grid, side).mean(axis=(1, 3)).reshape(-1)
    # Stable tie-breaking by row-major region ID is recorded for constant maps.
    order = np.argsort(importance, kind='stable')
    rng = np.random.default_rng(seed)
    selected = [('high', order[-count:][::-1].tolist()), ('low', order[:count].tolist())]
    selected += [('random', rng.choice(grid * grid, count, replace=False).tolist()) for _ in range(random_repeats)]
    return importance, selected


def occlusion_check(model, x, heatmap, target, *, baseline='mean', grid=7, count=5,
                    random_repeats=10, seed=42, batch_size=6):
    if x.shape != (1, 3, 224, 224) or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError('Expected one finite normalized RGB tensor of shape (1,3,224,224)')
    try:
        if isinstance(target, (bool, np.bool_)):
            raise TypeError
        target = operator.index(target)
    except TypeError as error:
        raise ValueError('Target must be an integer class index') from error
    if target not in range(4) or baseline not in {'mean', 'blur'} or batch_size < 1:
        raise ValueError('Invalid target, baseline or batch size')
    importance, selected = region_selections(heatmap, grid, count, random_repeats, seed)
    side = 224 // grid
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        with torch.inference_mode():
            original = model(x)[0]
            if original.shape != (4,) or not torch.isfinite(original).all():
                raise ValueError('Expected four finite logits')
            replacement = torch.zeros_like(x) if baseline == 'mean' else gaussian_blur(x, [31, 31], [10., 10.])
            variants = []
            for _, regions in selected:
                changed = x.clone()
                for region in regions:
                    row, column = divmod(region, grid)
                    ys, xs = slice(row * side, (row + 1) * side), slice(column * side, (column + 1) * side)
                    changed[:, :, ys, xs] = replacement[:, :, ys, xs]
                variants.append(changed[0])
            scores = torch.cat([model(torch.stack(variants[i:i+batch_size])) for i in range(0, len(variants), batch_size)])
            if not torch.isfinite(scores).all():
                raise ValueError('Non-finite occluded logits')
            original_logit = float(original[target].cpu())
            original_probability = float(original.softmax(0)[target].cpu())
            scores, probabilities = scores.cpu(), scores.softmax(1).cpu()
        rows = []
        for i, (control, regions) in enumerate(selected):
            rows.append({'control': control, 'random_repeat': i - 2 if control == 'random' else None,
                         'regions': regions, 'occluded_pixels': count * side * side,
                         'occluded_fraction': count / grid ** 2, 'target_class': target,
                         'original_logit': original_logit, 'occluded_logit': float(scores[i, target]),
                         'logit_drop': original_logit - float(scores[i, target]),
                         'original_probability': original_probability,
                         'occluded_probability': float(probabilities[i, target]),
                         'probability_drop': original_probability - float(probabilities[i, target]),
                         'occluded_predicted_class': int(scores[i].argmax())})
        return {'target_class': target, 'baseline': baseline, 'seed': seed, 'grid': grid,
                'region_count': count, 'random_repeats': random_repeats,
                'spatially_constant_map': bool(importance.max() == importance.min()),
                'region_importance': importance.tolist(), 'rows': rows}
    finally:
        for module, training in modes:
            module.training = training
