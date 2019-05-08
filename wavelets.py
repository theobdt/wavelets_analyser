import argparse
import numpy as np
import numpy.ma as ma
import os
import cv2
import matplotlib.pyplot as plt
import subprocess
from scipy.signal import find_peaks, argrelextrema
from scipy import ndimage as ndi
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
    level_max = max(levels)
    if level_max > 0:
        coeffs = pywt.wavedecn(frames, wavelet, level=level_max)
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


def new_local_maxima_3D(data, order=1):
    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    footprint[order, order, order] = 0
    filtered = ndi.maximum_filter(data, footprint=footprint)
    mask_local_maxima = data > filtered
    assert mask_local_maxima.dtype == bool
    sorted_abs_values = np.sort(np.abs(data[mask_local_maxima]))[::-1]
    local_max = ma.MaskedArray(data, mask=~mask_local_maxima)
    # mask : everything 0 becomes 0


def local_maxima_3D(data, order=1):
    """Detect local maxima of a 3D array

    Parameters
    ---------
    data : 3d ndarray
    order : int
        Number of neighbors used to compute local maxima

    Returns
    -------
    coordinates : ndarray
        coordinates of the local maxima
    values :
        values of the local maxima
    """
    peaks0 = np.array(argrelextrema(data, np.greater, axis=0, order=order))
    peaks1 = np.array(argrelextrema(data, np.greater, axis=1, order=order))
    peaks2 = np.array(argrelextrema(data, np.greater, axis=2, order=order))

    stacked = np.vstack((peaks0.transpose(), peaks1.transpose(),
                         peaks2.transpose()))

    elements, counts = np.unique(stacked, axis=0, return_counts=True)

    coords = elements[np.where(counts == 3)[0]]
    values = data[coords[:, 0], coords[:, 1], coords[:, 2]]

    return coords, values


def filter_coordinates(data, coordinates):
    filtered = np.zeros(data.shape)
    (i, j, k) = np.hsplit(coordinates, 3)
    filtered[i, j, k] = data[i, j, k]
    return filtered


def filter_local_maxima(coeffs, fraction_maxima=0.1):
    ordered_maxima = {}
    all_maxima = []

    coeffs[0] = np.zeros_like(coeffs[0])

    # step 1 : finding local maxima
    print('Finding local maxima ..')
    for i, level in enumerate(coeffs[1:]):
        print(f'level {i}')
        for coeff in list(level.keys()):
            print(f'coeff {coeff}')
            data = level[coeff]
            absolute = np.abs(data)
            print('local maxima')
            coords, abs_values = local_maxima_3D(absolute)
            # filtered = filter_coordinates(data, coords)
            # level[coeff] = filtered
            print('append')
            all_maxima += list(abs_values)
            print('stack')
            stacked = np.hstack((coords, abs_values.reshape(-1, 1)))
            ordered_maxima[(i, coeff)] = stacked

    # step 2 : selecting local maxima
    n_limit = int((1 - fraction_maxima) * len(all_maxima))
    limit = sorted(all_maxima)[n_limit]

    # step 3 : filtering local maxima
    print('Filtering local maxima ..')
    count_filtered_maxima = {}
    for i, level in enumerate(coeffs[1:]):
        print(f'level {i}')
        for coeff in list(level.keys()):
            print(f'coeff {coeff}')
            coeffs_values = ordered_maxima[(i, coeff)]
            mask = coeffs_values[:, -1] > limit
            if np.any(mask):
                count_filtered_maxima[(i, coeff)] = np.sum(mask)
                filtered_coords = coeffs_values[:, :-1][mask].astype(int)
                filtered_coeffs = filter_coordinates(level[coeff],
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
                        default='std',
                        help='Function used on reconstructed frames to get'
                             ' a 1D signal')
    parser.add_argument('-p', '--peaks',
                        type=float,
                        default=0.5,
                        help='Fraction of peaks selected, must be in [0-1]')
    parser.add_argument('-lm', '--local_maxima',
                        type=float,
                        help=('Fraction of local maxima selected, must be '
                              'in [0-1]'))
    args = parser.parse_args()

    # process video
    if not os.path.exists(args.input):
        print('Input path does not exist')
        return
    if args.signal not in ['mean', 'std']:
        print('Signal not recognized')
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
    else:
        dictionary = {args.level: [args.coeff]}

    frames, fps = import_frames(args.input)
    if args.local_maxima:
        reconstructed = reconstruct_local_maxima(frames, args.local_maxima,
                                                 args.level,
                                                 args.wavelet)
    else:
        print('Coefficients selected :')
        print(dictionary)
        reconstructed = reconstruct(frames, dictionary=dictionary,
                                    wavelet=args.wavelet)

    if args.signal == 'std':
        signal = np.std(reconstructed, axis=(1, 2))
    elif args.signal == 'mean':
        signal = np.mean(reconstructed, axis=(1, 2))

    # find peaks
    x_peaks = find_peaks(signal)[0]
    y_peaks = [signal[x] for x in x_peaks]
    sort = np.argsort(y_peaks)[::-1]
    selected = sort[: int(args.peaks * len(sort))]
    x_selected = [x_peaks[s] for s in selected]
    y_selected = [y_peaks[s] for s in selected]

    # plotting
    fig, ax = plt.subplots(3, figsize=(8, 8))
    ax[0].set_title('Gray')
    ax[1].set_title('Reconstructed')
    ax[2].set_title('Signal')

    filename = args.input.split('/')[-1]
    title = (f"filename={filename}, level/coeffs={dictionary}, p={args.peaks},"
             f" signal={args.signal}, wavelet={args.wavelet}")
    fig.suptitle(title, fontsize=13)
    im1 = ax[0].imshow(frames[0], plt.cm.gray)
    im2 = ax[1].imshow(reconstructed[0])
    ax[2].plot(signal)

    ymax = np.max(signal)
    ymin = np.min(signal)
    line, = ax[2].plot([0, 0], [ymax, ymin])
    ax[2].scatter(x_selected, y_selected, c='r', marker='X')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def update(i):
        i = i % frames.shape[0]
        im1.set_data(frames[i])
        im2.set_data(reconstructed[i])
        line.set_data([i, i], [ymax, ymin])

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
