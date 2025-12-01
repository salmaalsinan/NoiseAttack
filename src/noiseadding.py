import random
import pylops
from pylops.utils.wavelets import ricker
import numpy as np
from numpy.fft import fft, fftfreq, ifft
from scipy import fftpack

import cv2
from math import sqrt
from math import exp

from numpy import sqrt, newaxis, integer
from numpy.fft import irfft, rfftfreq
from numpy.random import default_rng, Generator, RandomState
from numpy import sum as npsum
from collections.abc import Iterable
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.signal import butter, lfilter, freqz
from utils import ScaleNormalize, RandomShift,calculate_psnr, snr_ratio


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='high', analog=False)

def butter_highpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5, **kwargs):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data, **kwargs)
    return y

def butter_highpass_filter(data, cutoff, fs, order=5, **kwargs):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data, **kwargs)
    return y


class BaseNoise():
    def __shifted_call__(self, sample):
        out = self.__call__(sample)
        return {'input': out['input'] - sample['input'], 'target': out['target']}


class add_gaussnoise(BaseNoise):
    def __init__(self, ampl_mu=3.0, ampl_std=1e-5):
        self.mu = ampl_mu
        self.std = ampl_std

    def __call__(self, sample):
        noiselevel = np.random.uniform(low=self.mu - sqrt(3) * self.std, high=self.mu + sqrt(3) * self.std)
        input, target = sample['input'], sample['target']
        input = input.copy()

        img = input.squeeze()
        noise = np.random.normal(loc=0, scale=noiselevel, size=img.shape) * 0.01 * noiselevel

        input += np.expand_dims(noise, axis=2)

        return {'input': input, 'target': target}


class add_color_noise(BaseNoise):
    def __init__(self, ampl_mu=0.005, ampl_std=1e-5):
        self.mu = ampl_mu
        self.std = ampl_std

    def powerlaw_psd_gaussian(self, exponent, size, fmin, random_state):
        # Make sure size is a list so we can iterate it and assign to it.
        try:
            size = list(size)
        except TypeError:
            size = [size]

        # The number of samples in each time series
        samples = size[-1]

        # Calculate Frequencies (we asume a sample rate of one)
        # Use fft functions for real output (-> hermitian spectrum)
        f = rfftfreq(samples)

        # Validate / normalise fmin
        if 0 <= fmin <= 0.5:
            fmin = max(fmin, 1. / samples)  # Low frequency cutoff
        else:
            raise ValueError("fmin must be chosen between 0 and 0.5.")

        # Build scaling factors for all frequencies
        s_scale = f
        ix = npsum(s_scale < fmin)  # Index of the cutoff
        if ix and ix < len(s_scale):
            s_scale[:ix] = s_scale[ix]
        s_scale = s_scale ** (-exponent / 2.)

        # Calculate theoretical output standard deviation from scaling
        w = s_scale[1:].copy()
        w[-1] *= (1 + (samples % 2)) / 2.  # correct f = +-0.5
        sigma = 2 * sqrt(npsum(w ** 2)) / samples

        # Adjust size to generate one Fourier component per frequency
        size[-1] = len(f)

        # Add empty dimension(s) to broadcast s_scale along last
        # dimension of generated random power + phase (below)
        dims_to_add = len(size) - 1
        s_scale = s_scale[(newaxis,) * dims_to_add + (Ellipsis,)]

        # prepare random number generator
        normal_dist = self._get_normal_distribution(random_state)

        # Generate scaled random power + phase
        sr = normal_dist(scale=s_scale, size=size)
        si = normal_dist(scale=s_scale, size=size)

        # If the signal length is even, frequencies +/- 0.5 are equal
        # so the coefficient must be real.
        if not (samples % 2):
            si[..., -1] = 0
            sr[..., -1] *= sqrt(2)  # Fix magnitude

        # Regardless of signal length, the DC component must be real
        si[..., 0] = 0
        sr[..., 0] *= sqrt(2)  # Fix magnitude

        # Combine power + corrected phase to Fourier components
        s = sr + 1J * si

        # Transform to real time series & scale to unit variance
        y = irfft(s, n=samples, axis=-1) / sigma

        return y

    def _get_normal_distribution(self, random_state):
        normal_dist = None
        if isinstance(random_state, (integer, int)) or random_state is None:
            random_state = default_rng(random_state)
            normal_dist = random_state.normal
        elif isinstance(random_state, (Generator, RandomState)):
            normal_dist = random_state.normal
        else:
            raise ValueError(
                "random_state must be one of integer, numpy.random.Generator, "
                "numpy.random.Randomstate"
            )
        return normal_dist

    def __call__(self, sample):
        noisescale = np.random.uniform(low=self.mu - sqrt(3) * self.std, high=self.mu + sqrt(3) * self.std)
        fmin = 0
        input, target = sample['input'], sample['target']
        input = input.copy()


        img = input.squeeze()
        im = []
        exponent = random.choice([1, 2])
        im.append(self.powerlaw_psd_gaussian(exponent, img.shape, fmin, None))
        y = np.array(np.array(im)).squeeze() * noisescale
        input += np.expand_dims(y, axis=2)

        return {'input': input, 'target': target}


class add_bandpassed_noise(BaseNoise):
    def __init__(self, ampl_mu=0.005, ampl_std=1e-5, flow_mu=1.5, flow_std=1e-3, fhigh_mu=5, fhigh_std=0.5):
        self.flow_mu = flow_mu
        self.flow_std = flow_std
        self.fhigh_mu = fhigh_mu
        self.fhigh_std = fhigh_std
        self.mu = ampl_mu
        self.std = ampl_std



    # USEFUL FUNCTIONS
    def band_limited_noise(self, min_freq, max_freq, np_seed_rnd, samples=1024, samplerate=1):
        freqs = np.fft.rfftfreq(samples, d=1 / samplerate)
        f = np.zeros(samples)
        idx = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
        f[idx] = 1
        return self.fftnoise(f, np_seed_rnd)

    def col_array_noise(self, datashape, np_seed_rnd, mnfreq=2, mxfreq=120, fs=500):
        noise_mod = np.zeros(datashape).T
        for i in range(len(noise_mod)):
            noise_mod[i, :] = self.band_limited_noise(mnfreq, mxfreq, np_seed_rnd, datashape[0], fs)
        return (noise_mod / np.mean(abs(noise_mod))).T

    def make_data(self, d, noiserange, nfreqrange, dt=500, nrels=500):
        noisy_data_list = []

        #     nsc = 0
        for i in range(nrels):
            nsc = np.random.uniform(low=self.mu - sqrt(3) * self.std, high=self.mu + sqrt(3) * self.std) * 0.1

            # Make bandpassed noise
            noise = self.col_array_noise(d.shape,
                                         np.random.RandomState(seed=0),
                                         mnfreq=nfreqrange[0],
                                         mxfreq=nfreqrange[1],
                                         fs=1 / dt
                                         )
            dn = d + (noise * nsc)
            noisy_data_list.append(np.expand_dims(np.expand_dims(dn, 0), 3))
        return noisy_data_list

    def fftnoise(self, f, np_seed_rnd):
        f = np.array(f, dtype='complex')
        Np = (len(f) - 1) // 2
        phases = np_seed_rnd.rand(Np) * 2 * np.pi
        phases = np.cos(phases) + 1j * np.sin(phases)
        f[1:Np + 1] *= phases
        f[-1:-1 - Np:-1] = np.conj(f[1:Np + 1])
        return np.fft.ifft(f).real

    def __call__(self, sample):
        f_low = np.random.uniform(low=self.flow_mu - sqrt(3) * self.flow_std,
                                  high=self.flow_mu + sqrt(3) * self.flow_std)
        f_high = np.random.uniform(low=self.fhigh_mu - sqrt(3) * self.fhigh_std,
                                   high=self.fhigh_mu + sqrt(3) * self.fhigh_std)

        input, target = sample['input'], sample['target']
        input = input.copy()
        img = input.squeeze()
        dt = 0.002
        freqrange = [f_low, f_high]
        y = np.array(self.make_data(np.zeros_like(img),
                                    [0.02, 0.35],
                                    freqrange,
                                    dt=dt,
                                    nrels=1)).squeeze()
        input += np.expand_dims(y, axis=2) # to make it similar to colored noise aka horizontal line with temp variation use y.T
        return {'input': input, 'target': target}


class add_blurnoise(BaseNoise):
    def __init__(self, ampl_mu=0.005, ampl_std=1e-5, kersize_mu=2, kersize_std=2e-4):
        self.mu = ampl_mu
        self.std = ampl_std
        self.kersize_mu = kersize_mu
        self.kersize_std = kersize_std

    def __call__(self, sample):
        ksizemax = int(np.random.uniform(low=self.kersize_mu - sqrt(3) * self.kersize_std,
                                         high=self.kersize_mu + sqrt(3) * self.kersize_std))

        noisescale = np.random.uniform(low=self.mu - sqrt(3) * self.std, high=self.mu + sqrt(3) * self.std)
        input, target = sample['input'], sample['target']
        input = input.copy()
        img = input.squeeze()
        imshape = img.shape
        ksize = (ksizemax, ksizemax)
        input = np.expand_dims((cv2.blur(img, ksize, cv2.BORDER_DEFAULT) - img) * noisescale + img, axis=2)
        return {'input': input, 'target': target}


def odd(l, u):
    return ([a for a in range(l, u) if a % 2 != 0])


class add_rainnoise(BaseNoise):
    def __init__(self, ampl_mu=0.005, ampl_std=1e-5):
        self.mu = ampl_mu
        self.std = ampl_std

    def generate_random_lines(self, imshape, slant, drop_length, rain_type):
        drops = []
        area = imshape[0] * imshape[1]
        no_of_drops = area // 600
     
        if rain_type.lower() == 'drizzle':
            no_of_drops = area // 770
            drop_length = 10
        elif rain_type.lower() == 'heavy':
            drop_length = 30
        elif rain_type.lower() == 'torrential':
            no_of_drops = area // 3500
            drop_length = 60
        for i in range(no_of_drops):  ## If You want heavy rain, try increasing this
            if slant < 0:
                x = np.random.randint(slant, imshape[1])
            else:
                x = np.random.randint(0, imshape[1] - slant)
            y = np.random.randint(0, imshape[0] - drop_length)

            drops.append((x, y))
        return drops, drop_length

    def rain_process(self, image, slant, drop_length, drop_color, drop_width, rain_drops):
        imshape = image.shape
        image_t = image.copy()
        for rain_drop in rain_drops:
            cv2.line(image_t, (rain_drop[0], rain_drop[1]), (rain_drop[0] + slant, rain_drop[1] + drop_length),
                     drop_color, drop_width)
        image = cv2.blur(image_t, (1, 1))  ## rainy view are blurry
        return image

    def __call__(self, sample):
        input, target = sample['input'], sample['target']
        input = input.copy()
        noisescale = np.random.uniform(low=self.mu - sqrt(3) * self.std, high=self.mu + sqrt(3) * self.std)
        img = input.squeeze()
        slant = -1

        drop_length = 10
        drop_width = 1
        drop_color = (1)  ## (200,200,200) a shade of gray

        slant_extreme = slant
        raintype = ['drizzle', 'heavy', 'torrential']
        imshape = img.shape
        slant = np.random.randint(-10, 10)  ##generate random slant if no slant value is given
        rain_type = 'torrential'
        rain_drops, drop_length = self.generate_random_lines(imshape, slant, drop_length, rain_type)
        drop_length = drop_length
        drop_width = drop_width
        drop_color = drop_color
        noise = self.rain_process(img, slant_extreme, drop_length, drop_color, drop_width, rain_drops)
        input = np.expand_dims((noise - input) * noisescale, axis=2)
        return {'input': input, 'target': target}


class add_spnoise(BaseNoise):
    def __init__(self, ampl_mu=0.005, ampl_std=1e-5, per_mu=10, per_std=2e-1):
        self.mu = ampl_mu
        self.std = ampl_std
        self.per_mu = per_mu
        self.per_std = per_std

    def __call__(self, sample):
        noisescale = np.random.uniform(low=self.mu - sqrt(3) * self.std, high=self.mu + sqrt(3) * self.std)

        percentage = np.random.uniform(low=self.per_mu - sqrt(3) * self.per_std,
                                       high=self.per_mu + sqrt(3) * self.per_std)

        input, target = sample['input'], sample['target']
        input = input.copy()
        img = input.squeeze()
        row, col = img.shape
        s_vs_p = 0.50
        amount = percentage / 100
        out = np.copy(img)
        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in img.shape]
        out[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in img.shape]
        out[coords] = 0
        input = np.expand_dims(((out - img) * noisescale + img), axis=2)
        return {'input': input, 'target': target}


class add_specklenoise(BaseNoise):
    def __init__(self, ampl_mu=0.005, ampl_std=1e-5):
        self.mu = ampl_mu
        self.std = ampl_std

    def __call__(self, sample):
        noisescale = np.random.uniform(low=self.mu - sqrt(3) * self.std, high=self.mu + sqrt(3) * self.std)

        input, target = sample['input'], sample['target']
        input = input.copy()
        img = input.squeeze()

        row, col = img.shape
        gauss = np.random.randn(row, col)
        gauss = gauss.reshape(row, col)
        noisy = img + img * gauss * noisescale
        input = np.expand_dims(noisy, axis=2)
        return {'input': input, 'target': target}


class add_noise_FFT(BaseNoise):
    def __init__(self, masktype="crossfilter", ampl_mu=0.005, ampl_std=1e-5, frac_mu=0.3, frac_std=1e-3, flow_mu=80,
                 flow_std=1e-3, fhigh_mu=10, fhigh_std=0.5, per_mu=5, per_std=2e-1):
        self.masktype = masktype
        self.mu = ampl_mu
        self.std = ampl_std

        self.flow_mu = flow_mu
        self.flow_std = flow_std
        self.fhigh_mu = fhigh_mu
        self.fhigh_std = fhigh_std

        self.per_mu = per_mu
        self.per_std = per_std
        self.frac_mu = frac_mu
        self.frac_std = frac_std

    def distance(self, point1, point2):
        return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def gaussianLP(self, D0, imgShape):
        base = np.zeros(imgShape[:2])
        rows, cols = imgShape[:2]
        center = (rows / 2, cols / 2)
        for x in range(cols):
            for y in range(rows):
                base[y, x] = exp(((-self.distance((y, x), center) ** 2) / (2 * (D0 ** 2))))
        return base

    def gaussianHP(self, D0, imgShape):
        base = np.zeros(imgShape[:2])
        rows, cols = imgShape[:2]
        center = (rows / 2, cols / 2)
        for x in range(cols):
            for y in range(rows):
                base[y, x] = 1 - exp(((-self.distance((y, x), center) ** 2) / (2 * (D0 ** 2))))
        return base

    def __call__(self, sample):
        keep_fraction = np.random.uniform(low=self.frac_mu - sqrt(3) * self.frac_std,
                                          high=self.frac_mu + sqrt(3) * self.frac_std)
        noiselevel = np.random.uniform(low=self.mu - sqrt(3) * self.std, high=self.mu + sqrt(3) * self.std)
        f_lowpas = np.random.uniform(low=self.flow_mu - sqrt(3) * self.flow_std,
                                     high=self.flow_mu + sqrt(3) * self.flow_std)
        f_high = np.random.uniform(low=self.fhigh_mu - sqrt(3) * self.fhigh_std,
                                   high=self.fhigh_mu + sqrt(3) * self.fhigh_std)
        percentage = np.random.uniform(low=self.per_mu - sqrt(3) * self.per_std,
                                       high=self.per_mu + sqrt(3) * self.per_std)

        input, target = sample['input'], sample['target']
        input = input.copy()
        img = input.squeeze()
        masktype = self.masktype

        im_fft = fftpack.fft2(img)
        if masktype == "crossfilter":
            im_fft2 = im_fft.copy()
            # Define the fraction of coefficients (in each direction) we keep
            keep_fraction = keep_fraction
            r, c = im_fft2.shape
            im_fft2[int(r * keep_fraction):int(r * (1 - keep_fraction))] = 0
            # # Similarly with the columns:
            im_fft2[:, int(c * keep_fraction):int(c * (1 - keep_fraction))] = 0
            # eal part for display.
            im_new = fftpack.ifft2(im_fft2).real

        elif masktype == "addrandomnoise":
            im_fft2 = im_fft.copy()
            im_fft2 = im_fft2 + np.random.normal(loc=0, scale=noiselevel, size=np.abs(im_fft2).shape)
            im_new = fftpack.ifft2(im_fft2).real

        elif masktype == "adds&pnoise":
            im_fft2 = im_fft.copy()
            speclist = []
            img = im_fft2
            row, col = img.shape
            s_vs_p = 0.5
            amount = percentage / 100
            out = np.copy(img)
            num_salt = np.ceil(amount * img.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in img.shape]
            out[coords] = 1
            num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in img.shape]
            out[coords] = 0
            im_new = fftpack.ifft2(out).real
        elif masktype == "lowpassfilter":
            center = np.fft.fftshift(im_fft)
            d0 = f_lowpas
  
            LowPassCenter = center * self.gaussianLP(d0, img.shape)
            im_fft2 = np.fft.ifftshift(LowPassCenter)
            im_new = fftpack.ifft2(im_fft2).real
        elif masktype == "highpassfilter":
            center = np.fft.fftshift(im_fft)
            d0 = f_high

            HighPassCenter = center * self.gaussianHP(d0, img.shape)
            im_fft2 = np.fft.ifftshift(HighPassCenter)
            im_new = fftpack.ifft2(im_fft2).real

        input = np.expand_dims(im_new, axis=2)
        return {'input': input, 'target': target}


class add_linearnoise(BaseNoise):
    def __init__(self, v_mu=110, v_std=1e-2,
                 tsample_mu=0.065, tsample_std=1e-3,
                 slope_mu=7.5, slope_std=1e-3,
                 ampl_mu=3.65, ampl_std=0.5):
        self.v_mu = v_mu
        self.v_std = v_std
        self.tsample_mu = tsample_mu
        self.tsample_std = tsample_std
        self.slope_mu = slope_mu
        self.slope_std = slope_std
        self.mu = ampl_mu
        self.std = ampl_std

    def __call__(self, sample):
        v = np.random.uniform(low=self.v_mu - sqrt(3) * self.v_std, high=self.v_mu + sqrt(3) * self.v_std)
        tsample = np.random.uniform(low=self.tsample_mu - sqrt(3) * self.tsample_std,
                                    high=self.tsample_mu + sqrt(3) * self.tsample_std)
        slope = np.random.uniform(low=self.slope_mu - sqrt(3) * self.slope_std,
                                  high=self.slope_mu + sqrt(3) * self.slope_std)
        ampli = np.random.uniform(low=self.mu - sqrt(3) * self.std, high=self.mu + sqrt(3) * self.std)

        input, target = sample['input'], sample['target']

        input = input.copy()
        img = input.squeeze()
        f = random.choice(range(5, 50, 1))
        par = {"ox": 0, "dx": 2000 / (img.shape[1]), "nx": img.shape[1] / 2, "ot": 0, "dt": 0.002, "nt": img.shape[0],
               "f0": f}
        t0 = list(np.arange(-10, 2.0, tsample))
        theta = [slope] * len(t0)
        amp = [ampli] * len(t0)
        # create axis
        taxis, taxis2, xaxis, yaxis = pylops.utils.seismicevents.makeaxis(par)

        # create wavelet
        wav = ricker(taxis[:41], f0=par["f0"])[0]
        #         wav = ricker(taxis, f0=par["f0"])[0]
        y = (
            pylops.utils.seismicevents.linear2d(xaxis, taxis, v, t0, theta, amp, wav)[1]
        )
        y = np.hstack([np.flip(y.T, axis=1)[:, :], y.T])
        input += np.expand_dims(y, axis=2)

        return {'input': input, 'target': target}


class add_hyperbolic_noise(BaseNoise):

    def __init__(self, v_mu=100, v_std=1e-1,
                 tsample_mu=0.05, tsample_std=1e-3,
                 ampl_mu=3.65, ampl_std=0.5):
        self.v_mu = v_mu
        self.v_std = v_std
        self.tsample_mu = tsample_mu
        self.tsample_std = tsample_std
        self.mu = ampl_mu
        self.std = ampl_std


    def dist_calc(self, r, s):
        ''' euclidean distance
        '''
        dx = r[0] - s[0]
        dy = r[1] - s[1]
        dz = r[2] - s[2]
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def comp_tt(self, dist, vel):
        ''' assume constant vel model and compute traveltimes
        '''
        return dist / vel

    def compute_tts(self, source, recs, vel):
        ''' compute travel times between a single source and array of receivers
        '''

        dist = [(lambda r: self.dist_calc(r, source))(r) for r in recs]
        tts = [(lambda d: self.comp_tt(d, vel))(d) for d in dist]

        return np.array(tts)

    def __call__(self, sample):
        noiselevel = np.random.uniform(low=self.mu - sqrt(3) * self.std, high=self.mu + sqrt(3) * self.std)
        v = np.random.uniform(low=self.v_mu - sqrt(3) * self.v_std, high=self.v_mu + sqrt(3) * self.v_std)
        tsample = np.random.uniform(low=self.tsample_mu - sqrt(3) * self.tsample_std,
                                    high=self.tsample_mu + sqrt(3) * self.tsample_std)

        input, target = sample['input'], sample['target']
        input = input.copy()
        img = input.squeeze()

        dt = 0.002  # time sampling

        dd = np.zeros_like(img)
  

        num_events = random.choice(range(10, 20, 1))

        z = np.sort(np.random.randint(low=6500, high=7000, size=(num_events)))
        v = np.sort(np.random.randint(low=50, high=60, size=(num_events)))
   
        for j in range(num_events):
            f = 40
         
            length = 0.4
            x_axis = np.arange(start=0,
                               stop=1000,
                               step=1000 / (img.shape[1]))
            y_axis = np.arange(start=0,
                               stop=1,
                               step=50)

            rx, ry = np.meshgrid(x_axis, y_axis)
            rz = 0
            recs = np.vstack([rx.flatten(), ry.flatten(), np.ones_like(rx.flatten()) * rz]).T

            # Compute the traveltimes between a source and all the receivers

            tts = self.compute_tts([500, 0, z[j]], recs, v[j])
            t_wav = np.arange(-length / 2, (length - dt) / 2, dt)
            wav = (1.0 - 2.0 * (np.pi ** 2) * (f ** 2) * (t_wav ** 2)) * np.exp(-(np.pi ** 2) * (f ** 2) * (t_wav ** 2))

            y = pylops.waveeqprocessing.marchenko.directwave(wav, tts, img.shape[0], dt, nfft=None, dist=None,
                                                             kind='2d', derivative=True)

            input += np.expand_dims(y, axis=2) * noiselevel

        return {'input': input, 'target': target}

class TraceMask(BaseNoise):
    def __init__(self, n_mu=5, n_std=1e-3, w_mu=2, w_std=1e-3):
        self.n_mu = n_mu
        self.n_std = n_std
        self.w_mu = w_mu
        self.w_std = w_std

    def __call__(self, sample):
        img = sample['input'].copy()
        wx, wy = img.shape[:2]
        n = np.random.uniform(low=self.n_mu - sqrt(3) * self.n_std, high=self.n_mu + sqrt(3) * self.n_std)
        n = round(n)
        for i in range(n):
            w = np.random.uniform(low=self.w_mu - sqrt(3) * self.w_std, high=self.w_mu + sqrt(3) * self.w_std)
            w = round(w)
            c = np.random.randint(wy)
            lidx = max(0,c - w)
            ridx = min(wx, c + w)
            img[:,lidx:ridx] = 0.0

        return {'input': img,
                'target': sample['target']}

class HighPassFilter(BaseNoise):
    def __init__(self, cutoff_mu=0.999, cutoff_std=1e-5, order=5, dt=0.002):
        self.cutoff_mu = cutoff_mu
        self.cutoff_std = cutoff_std
        self.order = order
        self.dt = dt
        self.f0 = 1/self.dt

    def __call__(self, sample):
        self.cutoff = np.random.uniform(low=self.cutoff_mu - sqrt(3) * self.cutoff_std, high=self.cutoff_mu + sqrt(3) * self.cutoff_std)
        img = sample['input']
        b, a = butter_highpass(self.f0*self.cutoff/2.0, self.f0, self.order)
        filtered = butter_highpass_filter(img, self.f0*self.cutoff/2.0, self.f0, self.order, axis=0)
        return {'input': filtered,
                'target': sample['target']}

class LowPassFilter(BaseNoise):
    def __init__(self, cutoff_mu=0.999, cutoff_std=1e-5, order=5, dt=0.002):
        self.cutoff_mu = cutoff_mu
        self.cutoff_std = cutoff_std
        self.order = order
        self.dt = dt
        self.f0 = 1/self.dt

    def __call__(self, sample):
        self.cutoff = np.random.uniform(low=self.cutoff_mu - sqrt(3) * self.cutoff_std, high=self.cutoff_mu + sqrt(3) * self.cutoff_std)
        img = sample['input']
        b, a = butter_lowpass(self.f0*self.cutoff/2.0, self.f0, self.order)
        filtered = butter_lowpass_filter(img, self.f0*self.cutoff/2.0, self.f0, self.order, axis=0)
        return {'input': filtered,
                'target': sample['target']}

    def plot(self, img):
        wx, wy= img.shape
        sp = fft(img, axis=0)
        N = wx
        n = np.arange(N)
        f0 = 1/self.dt
        T = N*self.dt
        freq = n / T
        b, a = butter_lowpass(f0*self.cutoff/2.0, f0, self.order)
        filtered = butter_lowpass_filter(img, f0*self.cutoff/2.0, f0, self.order, axis=0)
        sp_filtered = fft(filtered, axis=0)
        plt.figure(figsize=[15,5])
        plt.subplot(131)
        plt.imshow(img, cmap='seismic')
        plt.subplot(132)
        plt.plot(freq, np.abs(sp).mean(axis=1), c='r')
        plt.plot(freq, np.abs(sp_filtered).mean(axis=1), c='b')
        w, h = freqz(b, a, fs=f0, worN=8000)
        plt.plot(w, np.abs(h), 'b')
        plt.xlim(0,freq[N//2])
        plt.axvline(f0*self.cutoff/2.0, color='k')
        plt.plot(f0*self.cutoff/2.0, 0.5 * np.sqrt(2), 'ko')
        # plt.xlim(0)
        plt.subplot(133)
        plt.imshow(filtered, cmap='seismic')
        plt.show()


class LowPassFilter_random(BaseNoise):
    def __init__(self, noise_scale=0.05,cutoff_std=0.05, order=5, dt=0.009): #correct is dt=0.009


        self.noise_scale= noise_scale
        if self.noise_scale !=None:
            self.cutoff_mu = 0.9-(np.clip(self.noise_scale,0.05,2) - 0.05)*(0.9-0.1)/(3-0.05)
        else:
           self.cutoff_mu = np.random.uniform(low=0.1, high=0.9, size=1) # Define your cutoff frequency at random
           #print('cut off',self.cutoff_mu)
        self.cutoff_std = cutoff_std
        self.order = order
        self.dt = dt
        self.f0 = 1/self.dt
    
    def butter_lowpass(self,cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self,img, cutoff, fs, order=5, axis=0):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, img, axis=axis)
        return y,b,a

    def __call__(self, sample):
        
        self.cutoff = np.random.uniform(low=self.cutoff_mu - sqrt(3) * self.cutoff_std, high=self.cutoff_mu + sqrt(3) * self.cutoff_std)
        img = sample['input'].copy()
        filtered,_,_ = self.butter_lowpass_filter(img, self.f0*self.cutoff/2.0, self.f0, self.order, axis=0)
        return {'input': filtered,
                'target': sample['target']}

    def plot(self, img):
        img=img.squeeze().numpy().copy()
        try:
            wx, wy= img.shape
        except ValueError:
            B,wx, wy= img.shape
            v=np.random.choice(np.arange(B))
            img=img[v]
            wx, wy= img.shape
            print('batchsize>1, will display random image from batch')
        sp = fft(img, axis=0)
        N = wx
        n = np.arange(N)
        f0 = 1/self.dt
        T = N*self.dt
        freq = n / T
        self.cutoff = np.random.uniform(low=self.cutoff_mu - sqrt(3) * self.cutoff_std, high=self.cutoff_mu + sqrt(3) * self.cutoff_std)

        filtered,b,a = self.butter_lowpass_filter(img, f0*self.cutoff/2.0, f0, self.order, axis=0)

        sp_filtered = fft(filtered, axis=0)
        fig,ax=plt.subplots(figsize=[15, 5], ncols=4, nrows=1)
        ax[0].imshow(img, cmap='seismic',vmin=-1, vmax=1)
        print('original min/max/mean',img.min(),img.max(),img.mean())
        ax[0].set_title('Original',fontsize=12,fontweight='bold')
 
        ax[1].plot(freq, np.abs(sp).mean(axis=1), c='r', label='Original')
        ax[1].plot(freq, np.abs(sp_filtered).mean(axis=1), c='b',label='Filtered')
        w, h = freqz(b, a, fs=f0)
        ax[1].plot(w, np.abs(h), '-m',label='Low-pass Filter')
        ax[1].set_xlim(0, freq[N//2])
        ax[1].axvline(f0 * self.cutoff / 2.0, color='k',label='Cutoff F').set_linestyle('--')

        ax[1].legend(facecolor='none',edgecolor='none',fontsize=9,loc='upper left')
        ax[1].set_title('Frequency Spectrum',fontsize=12,fontweight='bold')
        ax[1].plot(f0 * self.cutoff / 2.0, 0.5 * np.sqrt(2), 'ko')
        ax[1].set_xlabel('Frequency (Hz)',fontsize=10,fontweight='bold')
        ax[1].set_ylabel('Amplitude',fontsize=10,fontweight='bold')
        
        ax[2].imshow(filtered, cmap='seismic',vmin=-1, vmax=1)
        print('filtered min/max/mean',np.min(filtered),np.max(filtered),np.mean(filtered))
        ax[2].set_title('Filtered',fontsize=12,fontweight='bold')
        ax[2].set_xlabel(f'PSNR ={np.round(calculate_psnr(filtered,img),3)} \\ \u03B2 ={np.round(snr_ratio(filtered,img),2)}',fontsize=10,fontweight='bold')
        
        im=ax[3].imshow(np.abs(img-filtered), cmap='jet', vmin=0, vmax=1)
        ax[3].set_title('Absolute Difference',fontsize=12,fontweight='bold')
        fig.colorbar(im, ax=ax[3], orientation='vertical', shrink=0.8)
        fig.tight_layout()
        fig.set_size_inches(w=11,h=3)
        plt.subplots_adjust(right=1.2)
        plt.show()



class CombinedTransforms():
    def __init__(self, *transforms, scale=1):
        self.transforms = transforms
        self.scale = scale
        if not isinstance(scale, Iterable):
            self.scale = [scale] * len(transforms)

    def __call__(self, sample):
     
        noises = [transform.__shifted_call__(sample)['input'] * scale for transform, scale in
                  zip(self.transforms, self.scale)]
       
        noise = sum(noises)
        return {'input': noise + sample['input'], 'target': sample['target']}


        
gaussian_noise = [add_gaussnoise(ampl_mu=9.5, ampl_std=1.0)]
color_noise = [add_color_noise(ampl_mu=0.95, ampl_std=2e-1)]
linear_noise = [add_linearnoise(v_mu=110, v_std=15,
                                tsample_mu=0.065, tsample_std=1e-2,
                                slope_mu=9.5, slope_std=5,
                                ampl_mu=3.75, ampl_std=0.5)]

blurnoise = [add_blurnoise(ampl_mu=9.5, ampl_std=1.0)]
bandpassed_noise = [add_bandpassed_noise(ampl_mu=9.5, ampl_std=1.0)]
spnoise = [add_spnoise(ampl_mu=9.5, ampl_std=1.0)]
specklenoise = [add_specklenoise(ampl_mu=9.5, ampl_std=1.0)]
hyperbolic_noise = [add_hyperbolic_noise(ampl_mu=5.5, ampl_std=1)]
cross_noise_FFT = [add_noise_FFT(masktype="crossfilter")]
random_noise_FFT = [add_noise_FFT(masktype="addrandomnoise", ampl_mu=325.0, ampl_std=60)]
gaussian_linear_noise = gaussian_noise + linear_noise
gaussian_color_noise = gaussian_noise + color_noise
gaussian_fft_noise = gaussian_noise + random_noise_FFT
gaussian_color_linear_noise = gaussian_noise + color_noise + linear_noise
gaussian_color_linear_fft_noise = gaussian_color_linear_noise + random_noise_FFT
gaussian_color_fft_noise = gaussian_color_noise + random_noise_FFT
linear_hyperbolic_noise = linear_noise + hyperbolic_noise
gaussian_color_linear_fft_hyperbolic_noise = gaussian_color_linear_fft_noise + hyperbolic_noise
low_pass_noise = [LowPassFilter(cutoff_mu=0.6, cutoff_std=0.05), ScaleNormalize('input')]
trace_noise = [TraceMask(n_mu=5, n_std=2, w_mu=8, w_std=3)]
color_bandpassed_noise = color_noise + bandpassed_noise
random_shift = [RandomShift(mean=2.5, std=1.0, shift_mean=20, shift_std=10), ScaleNormalize('input'), ScaleNormalize('target')]
color_linear_noise = color_noise + linear_noise
hyperbolic_fft_bandpassed_noise = hyperbolic_noise + random_noise_FFT + bandpassed_noise
hyperbolic_fft_linear_noise = hyperbolic_noise + random_noise_FFT + linear_noise

low_pass_noise_random=[LowPassFilter_random(noise_scale=None, cutoff_std=0.05), ScaleNormalize('input')]
noise_types = {
    -1: [],
    0 : {'linear' : gaussian_color_noise, 'nonlinear' : []},
    1 : {'linear' : gaussian_color_linear_noise, 'nonlinear' : []},
    2 : {'linear' : gaussian_color_linear_fft_noise, 'nonlinear' : []},
    3 : {'linear' : gaussian_color_linear_fft_hyperbolic_noise, 'nonlinear' : []},
    4 : {'linear' : gaussian_color_linear_noise, 'nonlinear' : low_pass_noise},
    5 : {'linear' : gaussian_color_linear_noise, 'nonlinear' : (trace_noise + low_pass_noise)},
    6 : {'linear' : gaussian_color_linear_noise, 'nonlinear' : trace_noise},
    7 : {'linear' : gaussian_noise, 'nonlinear' : []},
    8 : {'linear' : color_noise, 'nonlinear' : [] },
    9 : {'linear' : linear_noise, 'nonlinear' : [] }, 
    10 : {'linear' : random_noise_FFT, 'nonlinear' : [] }, 
    11 : {'linear' : hyperbolic_noise, 'nonlinear' : [] }, 
    12 : {'linear' : bandpassed_noise, 'nonlinear' : [] },
    13 : {'linear' : [], 'nonlinear' : low_pass_noise},
    14 : {'linear' : [], 'nonlinear' : trace_noise },
    15 : {'linear' : gaussian_color_fft_noise, 'nonlinear' : [] },
    16 : {'linear' : gaussian_fft_noise, 'nonlinear' : [] },
    17 : {'linear' : linear_hyperbolic_noise, 'nonlinear' : [] },
    18 : {'linear' : color_linear_noise, 'nonlinear' : [] },
    19 : {'linear' : gaussian_linear_noise, 'nonlinear' : [] },
    20 : {'linear' : color_bandpassed_noise, 'nonlinear' : [] },
    21 : {'linear' : hyperbolic_fft_bandpassed_noise, 'nonlinear' : [] },
    22 : {'linear' : hyperbolic_fft_linear_noise, 'nonlinear' : [] },
    220 : {'linear' : hyperbolic_fft_linear_noise, 'nonlinear' : low_pass_noise_random },
    30 : {'linear' : gaussian_color_linear_fft_hyperbolic_noise, 'nonlinear' : low_pass_noise},
    130 : {'linear' : [], 'nonlinear' : low_pass_noise_random}
}


def build_noise_transforms(noise_type, scale):
    if noise_type == -1:
        return noise_types[-1]
    noise_transforms = noise_types[noise_type]
    if len(noise_transforms['linear']) ==0:
        noise_level = scale
    else:
        noise_level = scale / sqrt(len(noise_transforms['linear']))

    return noise_transforms['nonlinear'] + [CombinedTransforms(*noise_transforms['linear'], scale=noise_level)]

def Noise_LP (noise_scale=0.0,cutoff_std=0.05):
    low_pass_noise_random=[LowPassFilter_random(noise_scale=noise_scale,cutoff_std=cutoff_std), ScaleNormalize('input')]
    noise_types2 = {
                -1: [],
                130 : {'linear' : [], 'nonlinear' : low_pass_noise_random}}
    return noise_types2

def build_noise_transforms2(noise_type, scale,cutoff_std=0.05):
    noise_types2=Noise_LP(noise_scale=scale,cutoff_std=cutoff_std)
    if noise_type == -1:
        return noise_types2[-1]
    noise_transforms = noise_types2[noise_type]
    if len(noise_transforms['linear']) ==0:
        noise_level = scale
    else:
        noise_level = scale / sqrt(len(noise_transforms['linear']))

    return noise_transforms['nonlinear'] + [CombinedTransforms(*noise_transforms['linear'], scale=noise_level)]

