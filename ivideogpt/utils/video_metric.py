import torch
from torch import nn
from tqdm import trange
import piqa
import lpips
import numpy as np
import scipy.linalg
from typing import Tuple
import scipy
from torch.cuda.amp import custom_fwd


def batch_forward(batch_size, input1, input2, forward, verbose=False):
    assert input1.shape[0] == input2.shape[0]
    return torch.cat([forward(input1[i: i + batch_size], input2[i: i + batch_size]) for i in trange(0, input1.shape[0], batch_size, disable=not verbose)], dim=0)


class Evaluator(nn.Module):
    def __init__(self, i3d_path=None, detector_kwargs=None, max_batchsize=None):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.lpips = lpips.LPIPS(net='vgg')
        self.psnr = piqa.PSNR(epsilon=1e-08, value_range=1.0, reduction='none')
        self.ssim = piqa.SSIM(window_size=11, sigma=1.5, n_channels=3, reduction='none')

        self.i3d_model = torch.jit.load(i3d_path).eval()
        self.max_batchsize = max_batchsize

    def compute_fvd(self, real_feature, gen_feature):
        if real_feature.num_items == 0 or gen_feature.num_items == 0:
            raise ValueError("No data to compute FVD")

        mu_real, sigma_real = real_feature.get_mean_cov()
        mu_gen, sigma_gen = gen_feature.get_mean_cov()

        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        return float(fid)

    def compute_fvd_from_raw_data(self, real_data=None, gen_data=None):

        detector_kwargs = dict(rescale=True, resize=True,
                               return_features=True)  # Return raw features before the softmax layer.

        mu_real, sigma_real = compute_feature_stats_for_dataset(self.i3d_model, detector_kwargs=detector_kwargs,
                                                                data=real_data).get_mean_cov()

        mu_gen, sigma_gen = compute_feature_stats_for_dataset(self.i3d_model, detector_kwargs=detector_kwargs,
                                                              data=gen_data).get_mean_cov()

        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)  # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        return float(fid)

    # custom_fwd: turn off mixed precision to avoid numerical instability during evaluation
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, video_1, video_2):
        # video_1: ground-truth
        # video_2: reconstruction or prediction

        if video_1.shape[0] < video_2.shape[0]:
            B, T, C, H, W = video_1.shape
            t = video_2.shape[0] // B
            video_1 = video_1.repeat([t, 1, 1, 1, 1])

            video_1 = video_1.reshape(-1, C, H, W)
            video_2 = video_2.reshape(-1, C, H, W)

            mse = self.mse(video_1, video_2).mean([1, 2, 3])
            psnr = self.psnr(video_1, video_2)
            ssim = self.ssim(video_1, video_2)
            if self.max_batchsize is not None and video_1.shape[0] > self.max_batchsize:
                lpips = batch_forward(
                    self.max_batchsize,
                    video_1 * 2 - 1, video_2 * 2 - 1,
                    lambda x1, x2: self.lpips(x1, x2).mean((1, 2, 3)),
                )
            else:
                lpips = self.lpips(video_1 * 2 - 1, video_2 * 2 - 1).mean((1, 2, 3))

            # get best of t predictions
            return (
                mse.reshape(t, B, T).mean(-1).min(0).values.mean(),
                psnr.reshape(t, B, T).mean(-1).max(0).values.mean(),
                ssim.reshape(t, B, T).mean(-1).max(0).values.mean(),
                lpips.reshape(t, B, T).mean(-1).min(0).values.mean(),
            )
        else:
            B, T, C, H, W = video_1.shape
            video_1 = video_1.reshape(B * T, C, H, W)
            video_2 = video_2.reshape(B * T, C, H, W)

            return (
                self.mse(video_1, video_2).mean(),
                self.psnr(video_1, video_2).mean(),
                self.ssim(video_1, video_2).mean(),
                self.lpips(video_1 * 2 - 1, video_2 * 2 - 1).mean(),
            )


@torch.no_grad()
def compute_feature_stats_for_dataset(detector, detector_kwargs, data=None):
    stats = FeatureStats(capture_mean_cov=True)

    for i in range(data.size(0)):
        # [batch_size, c, t, h, w]
        images = data[i].permute(0, 2, 1, 3, 4).contiguous()
        if images.shape[1] == 1:
            images = images.repeat([1, 3, *([1] * (images.ndim - 2))])
        features = detector(images, **detector_kwargs)
        stats.append_torch(features)

    return stats


class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x):
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov


def compute_fvd2(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)  # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    return float(fid)


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0)  # [d]
    sigma = np.cov(feats, rowvar=False)  # [d, d]

    return mu, sigma
