import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import subprocess
from matplotlib.animation import FuncAnimation
import pywt


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


def reconstruct(frames, level=1, coeff='daa', wavelet='haar'):
    """Filter and reconstruct video using wavelets

    Parameters
    ----------
    frames : ndarray
        input array of gray frames, shape: (nframes, height, width)
    level : int
        wavelets level
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
    if level > 0:
        print('Computing coefficients ..')
        coeffs = pywt.wavedecn(frames, wavelet, level=level)
        if coeff != 'aaa':
            coeffs[0] = np.zeros_like(coeffs[0])
        for i in range(2, len(coeffs)):
            for m in list(coeffs[i].keys()):
                coeffs[i][m] = np.zeros_like(coeffs[i][m])
        for m in list(coeffs[1].keys()):
            if m != coeff:
                coeffs[1][m] = np.zeros_like(coeffs[1][m])
        print('Reconstructing frames ..')
        rec = pywt.waverecn(coeffs, wavelet)
    else:
        rec = frames
    return rec


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
    parser.add_argument('-w', '--wavelet',
                        type=str,
                        default='haar',
                        help=("Wavelets type: haar, db2 .."))
    parser.add_argument('-o', '--output',
                        type=str,
                        default=None,
                        help='output path to save the gif')
    parser.add_argument('-s', '--signal',
                        type=str,
                        default='std',
                        help='Function used on reconstructed frames to get'
                             ' a 1D signal')
    args = parser.parse_args()

    # process video
    if args.signal not in ['mean', 'std']:
        print('Signal not recognized')
        return
    if args.output is not None:
        save = True
    else:
        save = False

    frames, fps = import_frames(args.input)
    reconstructed = reconstruct(frames, wavelet=args.wavelet, level=args.level,
                                coeff=args.coeff)
    if args.signal == 'std':
        signal = np.std(reconstructed, axis=(1, 2))
    elif args.signal == 'mean':
        signal = np.mean(reconstructed, axis=(1, 2))

    # plotting
    fig, ax = plt.subplots(3, figsize=(8, 8))
    ax[0].set_title('Gray')
    ax[1].set_title('Reconstructed')
    ax[2].set_title('Signal')

    filename = args.input.split('/')[-1]
    title = (f"filename={filename}, level={args.level}, coeff={args.coeff}, "
             f"signal={args.signal}, wavelet={args.wavelet}")
    fig.suptitle(title, fontsize=13)
    im1 = ax[0].imshow(frames[0], plt.cm.gray)
    im2 = ax[1].imshow(reconstructed[0])
    ax[2].plot(signal)

    ymax = np.max(signal)
    ymin = np.min(signal)
    line, = ax[2].plot([0, 0], [ymax, ymin])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def update(i):
        i = i % frames.shape[0]
        im1.set_data(frames[i])
        im2.set_data(reconstructed[i])
        line.set_data([i, i], [ymax, ymin])

    if save:
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
