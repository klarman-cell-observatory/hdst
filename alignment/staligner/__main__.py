import argparse as ap

import logging

from . import align


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__package__)


def main():
    """Script entry point.
    """
    opt = ap.ArgumentParser()
    opt.add_argument('-i', '--input', type=str, required=True,
                     help='Input image.')
    opt.add_argument('-o', '--output-directory', type=str, default='.',
                     help='Output directory.')
    opt.add_argument('--size', nargs=2, default=[-1, 7330], type=int,
                     help='Rescale the image to at most the given size before '
                          'running.')
    opt.add_argument('--win-size', default=1 / 25, type=float,
                     help='Size of the decision window. Use smaller values if '
                     'the image is very rotated or if the tissue extends '
                     'close to the borders of the array.')
    opt.add_argument('--annotate', action='store_true',
                     help='Emit bright-field image with spot annotations.')
    opt.add_argument('--debug', action='store_true',
                     help='Print debug messages.')
    opt = opt.parse_args()

    if opt.debug:
        LOG.setLevel(logging.DEBUG)

    LOG.info('Running frame detection with options: %s.',
             ', '.join([f'{k}={v}' for k, v in vars(opt).items()]))
    align(
        im_file=opt.input,
        im_size=opt.size,
        win_size=[
            round(min([s for s in opt.size if s > 0]) * opt.win_size)] * 2,
        annotate=opt.annotate,
        output_directory=opt.output_directory,
    )


if __name__ == "__main__":
    main()
