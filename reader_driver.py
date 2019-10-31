from plate_reader import OpenALPRReader
import glob
import os
import sys

def process_image(imgpath, outdir):
    def get_basename(fname):
        base = os.path.basename(fname)
        return '.'.join(base.split('.')[:-1])

    with open('key.txt') as f:
        skey = f.read().strip()

    basename = get_basename(imgpath)
    reader = OpenALPRReader(skey)
    reader.process_image(
            imgpath, os.path.join(outdir, basename + '.json'))

def process_directory(indir, outdir):
    for imgpath in glob.glob(os.path.join(indir, '*.png')):
        process_image(imgpath, outdir)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("`Usage: python %s indir outdir`" % __file__)
        exit(1)
    else:
        _, indir, outdir, *_ = sys.argv
        process_directory(indir, outdir)
