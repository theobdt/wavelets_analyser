# Wavelets video analyser


## Installation

```
git clone https://github.com/theobdt/wavelets_analyser.git
cd wavelets_analyser
pip install -r requirements.txt
```
You need to have ffmpeg installed for this repo to work correctly.
```
sudo apt install ffmpeg
```

## Usage 
```
$ python wavelets.py -f <input_path> [-l <level>] [-c <coeff>] [-w <wavelet>] [-s <signal>] [-o <output_path>]
```

### Options

* -i, --input : input video path
* -l, --level : decomposition level, must be >= 0
* -c, --coeff : one of the 8 coefficients to choose within the current decomposition level: ['aaa', 'daa'(default), 'ada', 'aad', 'dda', 'add', 'ddd']
    a : approximation
    d : details
    order : (time, y, x) 
* -w, --wavelets : wavelet used (default='haar'), full list of wavelets available [here](https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html)
* -s, --signal : operation used to compress 3D reconstructed frames to 1D array: 'std' (default) or 'mean'
* -o, --output : output path. If specified, the animation is not drawn but is saved to this path.

## Examples
To visualize the animation:
```
$ python wavelets.py -i input_video.mp4
```
or 
```
$ python wavelets.py -i input_video.mp4 -l 2 -c ddd -s mean -w db2
```

To save the animation:
```
$ python wavelets.py -i input_video.mp4 -o output_animation.mp4
```
