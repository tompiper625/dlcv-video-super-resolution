"""Run a CPU pilot for single-neighbor occlusion in 2x video super-resolution."""

import csv
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage import data, io
from skimage.color import rgb2gray
from skimage.registration import optical_flow_tvl1
from skimage.transform import resize


TRAIN = ('astronaut.png', 'brick.png', 'camera.png', 'cell.png', 'clock_motion.png',
         'coins.png', 'grass.png', 'hubble_deep_field.jpg', 'motorcycle_left.png', 'page.png')
TEST = ('chelsea.png', 'coffee.png', 'gravel.png', 'ihc.png', 'moon.png',
        'retina.jpg', 'rocket.jpg', 'text.png')
METHODS = ('single', 'unaligned', 'aligned', 'confidence', 'oracle_mask')
SIDE = 128
MARGIN = 16
TRIM = 8
DISPLACEMENTS = (3, 15)


def load_image(name):
    image = io.imread(Path(data.data_dir) / name)
    if image.ndim == 3:
        return rgb2gray(image[..., :3]).astype(np.float32)
    return image.astype(np.float32) / np.iinfo(image.dtype).max


def sample_scene(image, rng):
    y, x = (int(rng.integers(MARGIN, size - SIDE - MARGIN + 1)) for size in image.shape)
    axis = int(rng.integers(2))
    sign = int(rng.choice((-1, 1)))
    blocked_neighbor = int(rng.choice((0, 2)))
    y_mask, x_mask = rng.integers(24, 49, size=2)
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    mask[y_mask:y_mask + 48, x_mask:x_mask + 48] = True
    noise = gaussian_filter(rng.standard_normal((SIDE, SIDE)), 2)
    color = rng.choice((0.12, 0.88))
    occluder = np.clip(color + 0.4 * noise, 0, 1).astype(np.float32)
    return y, x, axis, sign, blocked_neighbor, mask, occluder


def render(image, scene, shift, blocked):
    y, x, axis, sign, blocked_neighbor, mask, occluder = scene
    frames = []
    for number, position in enumerate((-1, 0, 1)):
        offset = position * sign * shift
        dy, dx = (offset, 0) if axis == 0 else (0, offset)
        frame = image[y + dy:y + dy + SIDE, x + dx:x + dx + SIDE].copy()
        if blocked and number == blocked_neighbor:
            shifted_mask = np.roll(mask, (-dy, -dx), axis=(0, 1))
            shifted_texture = np.roll(occluder, (-dy, -dx), axis=(0, 1))
            frame[shifted_mask] = shifted_texture[shifted_mask]
        frames.append(frame)
    low = tuple(resize(frame, (SIDE // 2,) * 2, order=3, anti_aliasing=True)
                for frame in frames)
    return low, frames[1]


def warp_neighbor(center, neighbor):
    forward = optical_flow_tvl1(center, neighbor, num_warp=3, num_iter=8)
    backward = optical_flow_tvl1(neighbor, center, num_warp=3, num_iter=8)
    coordinates = np.indices(center.shape, dtype=np.float32) + forward
    valid = ((coordinates >= 0) & (coordinates <= np.array(center.shape)[:, None, None] - 1)).all(axis=0)
    warped = map_coordinates(neighbor, coordinates, order=1, mode='nearest')
    inverse = np.stack([map_coordinates(field, coordinates, order=1, mode='nearest')
                        for field in backward])
    error = np.linalg.norm(forward + inverse, axis=0)
    weights = np.where(valid, np.exp(-error**2), 0)
    return np.where(valid, warped, center), weights


def inputs(low, scene, blocked):
    previous, center, following = low
    before, weight_before = warp_neighbor(center, previous)
    after, weight_after = warp_neighbor(center, following)
    gated = (center + weight_before * (before - center), center,
             center + weight_after * (after - center))
    oracle = [before, center, after]
    if blocked:
        mask = resize(scene[5], center.shape, order=0, anti_aliasing=False)
        index = scene[4]
        oracle[index] = np.where(mask, center, oracle[index])
    return ((center, center, center), low, (before, center, after), gated, oracle)


def features(frames):
    # A shared 5x5 linear predictor with four outputs arranged as a 2x2 HR block.
    packed = [sliding_window_view(np.pad(frame, 2, mode='reflect'), (5, 5))
              .reshape(frame.size, 25) for frame in frames]
    return np.column_stack((*packed, np.ones(frames[0].size, dtype=np.float32)))


def target_pixels(image):
    return image.reshape(SIDE // 2, 2, SIDE // 2, 2).transpose(0, 2, 1, 3).reshape(-1, 4)


def predict(frames, weights):
    predicted = (features(frames) @ weights).reshape(SIDE // 2, SIDE // 2, 2, 2)
    return np.clip(predicted.transpose(0, 2, 1, 3).reshape(SIDE, SIDE), 0, 1)


def cases(names, seed):
    rng = np.random.default_rng(seed)
    for name in names:
        image = load_image(name)
        for crop in range(3):
            scene = sample_scene(image, rng)
            for shift in DISPLACEMENTS:
                for blocked in (False, True):
                    low, target = render(image, scene, shift, blocked)
                    yield name, crop, shift, blocked, scene[5], target, inputs(low, scene, blocked)


def train():
    grams = np.zeros((len(METHODS), 76, 76))
    targets = np.zeros((len(METHODS), 76, 4))
    for _, _, _, _, _, target, sample in cases(TRAIN, 41):
        for index, frames in enumerate(sample):
            x = features(frames)
            grams[index] += x.T @ x
            targets[index] += x.T @ target_pixels(target)
    regularizer = 2 * np.eye(76)
    regularizer[-1, -1] = 0
    return [np.linalg.solve(gram + regularizer, target).astype(np.float32)
            for gram, target in zip(grams, targets)]


def evaluate(weights):
    crop = np.s_[TRIM:-TRIM, TRIM:-TRIM]
    for source, index, shift, blocked, mask, target, sample in cases(TEST, 73):
        for method, model, frames in zip(METHODS, weights, sample):
            residual = predict(frames, model)[crop] - target[crop]
            absolute = np.abs(residual)
            region = mask[crop]
            yield dict(source=source, crop=index, shift_lr=shift / 2, occluded=int(blocked),
                       method=method, psnr=-10 * np.log10(np.mean(residual**2)),
                       region_mae=absolute[region].mean(), outside_mae=absolute[~region].mean())


def summarize(rows):
    print('Condition            Model          PSNR      Region MAE   Outside MAE')
    for shift in DISPLACEMENTS:
        for blocked in (0, 1):
            paired = {}
            for method in METHODS:
                selected = [r for r in rows if r['shift_lr'] == shift / 2
                            and r['occluded'] == blocked and r['method'] == method]
                means = [np.mean([r[key] for r in selected])
                         for key in ('psnr', 'region_mae', 'outside_mae')]
                print(f'{shift / 2:3.1f}px / occluded={blocked}   {method:12}   '
                      f'{means[0]:6.2f}    {means[1]:8.5f}     {means[2]:8.5f}')
                paired[method] = {(r['source'], r['crop']): r['region_mae'] for r in selected}
            differences = [np.mean([paired['confidence'][source, crop_id]
                                    - paired['aligned'][source, crop_id] for crop_id in range(3)])
                           for source in TEST]
            bootstrap = np.random.default_rng(9).choice(differences, (5000, len(TEST))).mean(axis=1)
            lower, upper = np.percentile(bootstrap, (2.5, 97.5))
            print(f'  confidence - aligned region MAE: {np.mean(differences):+.5f}, '
                  f'95% source bootstrap [{lower:+.5f}, {upper:+.5f}]')


if __name__ == '__main__':
    print(f'Training on {len(TRAIN)} images, testing on {len(TEST)} held-out images.', flush=True)
    result = list(evaluate(train()))
    with Path(__file__).with_name('results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=result[0])
        writer.writeheader()
        writer.writerows(result)
    summarize(result)
