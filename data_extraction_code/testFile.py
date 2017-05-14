import numpy as np

import pyradi.ryptw as ryptw
import pyradi.ryplot as ryplot
import pyradi.ryfiles as ryfiles

from IPython.display import display
from IPython.display import Image
from IPython.display import HTML

#make pngs at 150 dpi
import matplotlib as mpl
mpl.rc("savefig", dpi=75)
mpl.rc('figure', figsize=(10,8))

tgzFilename = 'pyradiSamplePtw.tgz'
destinationDir = '.'
tarFilename = 'pyradiSamplePtw.tar'
url = 'https://raw.githubusercontent.com/NelisW/pyradi/master/pyradi/data/'
dlNames = ryfiles.downloadUntar(tgzFilename, url, destinationDir, tarFilename)
print('filesAvailable are {}'.format(dlNames))

# first read the ptw file
ptwfile  = './PyradiSampleLWIR.ptw'
outfilename = 'PyradiSampleLWIR.txt'

header = ryptw.readPTWHeader(ptwfile)
ryptw.showHeader(header)
