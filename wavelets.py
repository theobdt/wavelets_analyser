import argparse
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import subprocess
import librosa
import librosa.display
from scipy.signal import find_peaks
from scipy import ndimage as ndi
from skimage import filters
from matplotlib.animation import FuncAnimation
import pywt
import struct


def import_frames(path):
    """Creates gray frames array from input path

    Parameters:
    -----------
    path : str
        path of the input video

    Returns
    -------
    frames : ndarray
        gray frames
    fps : float
        frame rate of input video
    """
    print(f'Importing frames from {path} ..')
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    counter = 0
    frames = []
    while(cap.isOpened()):
        counter += 1
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    return np.array(frames), fps


def reconstruct(frames, dictionary, wavelet='haar'):
    """Filter and reconstruct video using wavelets

    Parameters
    ----------
    frames : ndarray
        input array of gray frames, shape: (nframes, height, width)
    dictionary : dict
        levels and coeffs to keep

        coeff : str
        Selects one of the 8 subsets of the 3D wavelets coefficient "cube",
        corresponding to the chosen wavelet level.
        3 characters long string, ex: 'aaa', 'aad', 'ada', 'add', ...
        a : approximation
        d : details
        order : (time, y, x)

    Returns
    -------
    rec : ndarray
        reconstructed frames from the chosen wavelet level and coeff
    """
    levels = list(dictionary.keys())
    print(levels)
    level_max = max(levels)
    if level_max > 0:
        coeffs = pywt.wavedecn(frames, wavelet, level=level_max)
        print(len(coeffs))
        print(coeffs[1].keys())
        for i in range(len(coeffs)):
            level = level_max - i + 1
            if i == 0:
                if 0 not in levels:
                    coeffs[0] = np.zeros_like(coeffs[0])
                continue
            for key in list(coeffs[i].keys()):
                if level in levels:
                    if key in dictionary[level]:
                        continue
                coeffs[i][key] = np.zeros_like(coeffs[i][key])
        print('Reconstructing frames ..')
        rec = pywt.waverecn(coeffs, wavelet)
    else:
        rec = frames
    return rec


def local_maxima_3D(data, order=1):
    """Detect local maxima of a 3D array

    Parameters
    ----------
    data : 3d ndarray
    order : int
        Number of neighbors used to compute local maxima

    Returns
    -------
    coordinates : ndarray
        coordinates of the local maxima (nmax, 3)
    values :
        values of the local maxima
    """
    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    footprint[order, order, order] = 0
    filtered = ndi.maximum_filter(data, footprint=footprint)
    mask_local_maxima = data > filtered
    assert mask_local_maxima.dtype == bool

    coords = np.asarray(np.where(mask_local_maxima)).T
    values = data[mask_local_maxima]

    return coords, values


def filter_coordinates(data, coordinates):
    """Set to zero points not listed in coordinates

    Parameters
    ----------
    data : 3d ndarray
        array to filter
    coordinates : ndarray (N, 3)
        3D coordinates of the N points to keep

    Returns
    -------
    filtered : ndarray (data.shape)
        array with filtered coordinates
    """
    filtered = np.zeros(data.shape)
    (i, j, k) = np.hsplit(coordinates, 3)
    filtered[i, j, k] = data[i, j, k]
    return filtered


def filter_local_maxima(coeffs, fraction_maxima=0.1):
    """Keep the first ($fraction_maxima)*100 percents of local maxima in coeffs

    Parameters
    ----------
    coeffs : dict
        wavelets coefficients
    fraction_maxima : float in [0-1]

    Returns
    -------
    coeffs : dict
        filtered coefficients
    """
    ordered_maxima = {}
    all_maxima = []

    coeffs[0] = np.zeros_like(coeffs[0])
    level_max = len(coeffs) - 1

    # step 1 : finding local maxima
    print('Finding local maxima ..')
    for i, group in enumerate(coeffs[1:]):
        level = level_max - i
        for coeff in list(group.keys()):
            print(f'level {level}, coefficients {coeff}')
            data = group[coeff]
            absolute = np.abs(data)
            coords, abs_values = local_maxima_3D(absolute)
            all_maxima += list(abs_values)
            stacked = np.hstack((coords, abs_values.reshape(-1, 1)))
            ordered_maxima[(i, coeff)] = stacked

    # step 2 : selecting local maxima
    n_limit = int((1 - fraction_maxima) * len(all_maxima))
    limit = sorted(all_maxima)[n_limit]

    # step 3 : filtering local maxima
    print('Filtering local maxima ..')
    count_filtered_maxima = {}
    for i, group in enumerate(coeffs[1:]):
        level = level_max - i
        for coeff in list(group.keys()):
            print(f'level {level}, coefficients {coeff}')
            coeffs_values = ordered_maxima[(i, coeff)]
            mask = coeffs_values[:, -1] > limit
            if np.any(mask):
                count_filtered_maxima[(i, coeff)] = np.sum(mask)
                filtered_coords = coeffs_values[:, :-1][mask].astype(int)
                filtered_coeffs = filter_coordinates(group[coeff],
                                                     filtered_coords)
                coeffs[i + 1][coeff] = filtered_coeffs
            else:
                count_filtered_maxima[(i, coeff)] = 0
                coeffs[i + 1][coeff] = np.zeros_like(coeffs[i + 1][coeff])

    print(count_filtered_maxima)
    return coeffs


def reconstruct_local_maxima(frames, fraction_maxima=0.1, level_max=3,
                             wavelet='haar'):
    print('Decomposing frames ..')
    coeffs = pywt.wavedecn(frames, wavelet, level=level_max)
    filtered_coeffs = filter_local_maxima(coeffs, fraction_maxima)
    print('Reconstructing frames ..')
    reconstructed = pywt.waverecn(filtered_coeffs, wavelet)
    return reconstructed


def segmentation_1d(signal, size=10):
    absolute_val = np.abs(signal)
    maxs = ndi.maximum_filter1d(absolute_val, size)
    zeros = np.zeros(len(maxs))
    if np.all(zeros == maxs):
        return zeros
    else:
        thresh = filters.threshold_otsu(maxs)
        binary = maxs > thresh
    return binary


def post_processing(frames):
    print('Post processing ..')
    post_processed = []
    for frame in frames:
        m = np.mean(frame)
        post_processed.append(frame - m)

    return np.asarray(post_processed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        type=str,
                        help='Path to the input video')
    parser.add_argument('-l', '--level',
                        type=int,
                        default=1,
                        help=('Wavelets decomposition '
                              'level (low frequencies)'))
    parser.add_argument('-c', '--coeff',
                        type=str,
                        default='daa',
                        help=("Wavelets coeffs ['aaa', daa' , 'ddd', ...] "
                              "a: approximation, d: details"))
    parser.add_argument('-t', '--txt_file',
                        type=str,
                        help='Text file with level and coeff decomposition')
    parser.add_argument('-w', '--wavelet',
                        type=str,
                        default='haar',
                        help=("Wavelets type: haar, db2 .."))
    parser.add_argument('-o', '--output',
                        type=str,
                        help='output path to save the animation')
    parser.add_argument('-sf', '--sound_file',
                        type=str,
                        help='output path to save the signal as sound')
    parser.add_argument('-s', '--signal',
                        type=str,
                        default='mean',
                        help='Function used on reconstructed frames to get'
                        ' a 1D signal: mean, std, mean_abs')
    parser.add_argument('-p', '--peaks',
                        type=float,
                        default=0,
                        help='Fraction of peaks selected, must be in [0-1]')
    parser.add_argument('-lm', '--local_maxima',
                        type=float,
                        help=('Fraction of local maxima selected, must be '
                              'in [0-1]'))
    parser.add_argument('-so', '--sound',
                        action='store_true',
                        help=('If entered, display sound waveform from '
                              'input file'))
    parser.add_argument('-pp', '--post_processing',
                        action='store_true',
                        help='If entered, process frames after reconstruction')
    args = parser.parse_args()

    # process video
    if not os.path.exists(args.input):
        print('Input path does not exist')
        return
    if args.signal not in ['mean', 'std', 'mean_abs']:
        print('Signal not recognized: mean, std, mean_abs')
        return
    if args.peaks < 0 or args.peaks > 1:
        print('Peaks fraction must be between 0 and 1')
        return
    if args.local_maxima:
        if args.local_maxima < 0 or args.local_maxima > 1:
            print('Fraction of local maxima must be between 0 and 1')
            return

    if args.txt_file:
        lines = []
        print(f'Reading coeffs from {args.txt_file} ..')
        with open(args.txt_file, 'r') as txt:
            line = txt.readline()
            while line:
                lines.append(line)
                line = txt.readline()
        dictionary = {}
        for line in lines:
            split = line.split()
            dictionary[int(split[0])] = split[1:]
        print(dictionary)
    else:
        dictionary = {args.level: [args.coeff]}

    frames, fps = import_frames(args.input)
    t_max = frames.shape[0] / fps
    if args.local_maxima:
        reconstructed = reconstruct_local_maxima(frames, args.local_maxima,
                                                 args.level,
                                                 args.wavelet)
    else:
        print('Coefficients selected :')
        print(dictionary)
        reconstructed = reconstruct(frames, dictionary=dictionary,
                                    wavelet=args.wavelet)
    if args.post_processing:
        reconstructed = post_processing(reconstructed)

    if args.signal == 'std':
        signal = np.std(reconstructed, axis=(1, 2))
    elif args.signal == 'mean':
        signal = np.mean(reconstructed, axis=(1, 2))
    elif args.signal == 'mean_abs':
        signal = np.mean(np.abs(reconstructed), axis=(1, 2))
    if np.all(signal == 0):
        print('WARNING : signal is equal to zero')
        print("Did you use 'mean' on detailed coefficients only ?")
    signal = signal / np.max(signal)
    # find peaks
    x_peaks = find_peaks(signal)[0]
    y_peaks = [signal[x] for x in x_peaks]
    sort = np.argsort(y_peaks)[::-1]
    selected = sort[: int(args.peaks * len(sort))]
    x_selected = [x_peaks[s] for s in selected]
    y_selected = [y_peaks[s] for s in selected]

    # segmentation 1d
    absolute = np.abs(signal)
    binary = segmentation_1d(absolute)
    scaled = np.array(binary) * np.max(absolute)

    # plotting
    fig, ax = plt.subplots(3, figsize=(8, 8))
    ax[0].set_title('Gray')
    ax[1].set_title('Reconstructed')
    ax[2].set_title('Signal')

    filename = args.input.split('/')[-1]
    title = (f"filename={filename}"
             f" signal={args.signal}, wavelet={args.wavelet}")
    fig.suptitle(title, fontsize=13)
    im1 = ax[0].imshow(frames[0], plt.cm.gray)
    im2 = ax[1].imshow(reconstructed[0])
    n_points = len(signal)
    times = np.linspace(0, t_max, n_points)

    ax[2].plot(times, signal, label='wavelet signal')
    # ax[2].plot(times, scaled)

    ymax = np.max(signal)
    ymin = np.min(signal)
    line, = ax[2].plot([0, 0], [ymax, ymin])
    ax[2].scatter(x_selected, y_selected, c='r', marker='X')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if args.sound:
        y, sr = librosa.load(args.input)
        y = y / np.max(y)
        librosa.display.waveplot(y, sr=sr, ax=ax[2], label='audio waveform')

    def update(i):
        i = i % len(signal)
        im1.set_data(frames[i])
        im2.set_data(reconstructed[i])
        x_line = times[i]
        line.set_data([x_line, x_line], [ymax, ymin])

    if args.sound_file:
        m = np.max(np.abs(signal))
        print(f'Saving audio to {args.sound_file}')
        with open(args.sound_file, 'wb') as sfile:
            for y in signal:
                sfile.write(struct.pack('b', int(y*127/m)))

    if args.output:
        print(f'Saving output ..')
        # Open an ffmpeg process
        outf = args.output
        canvas_width, canvas_height = fig.canvas.get_width_height()
        cmdstring = ('ffmpeg',
                     '-y', '-r', f'{fps}',
                     '-s', f'{canvas_width}x{canvas_height}',
                     '-pix_fmt', 'argb',
                     '-f', 'rawvideo',  '-i', '-',
                     '-vcodec', 'libx264', outf)
        p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

        for i in range(frames.shape[0]):
            update(i)
            fig.canvas.draw()
            string = fig.canvas.tostring_argb()
            # write to pipe
            p.stdin.write(string)

        p.communicate()
        print(f'Output saved to {args.output}')
    else:
        print('Displaying animation ..')
        anim = FuncAnimation(fig, update, frames=np.arange(frames.shape[0]))
        plt.show()
        print('Animation closed')


if __name__ == '__main__':
    main()
