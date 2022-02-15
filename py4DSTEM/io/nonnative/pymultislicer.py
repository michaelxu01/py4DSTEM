# Reads an EMPAD file
#
#
# @author: mxu86

import numpy as np
from pathlib import Path
from ..datastructure import DataCube
from ...process.utils import bin2D, tqdmnd


def read_pymultislicer(filename, mem="RAM", binfactor=1, metadata=False, **kwargs):
    """
    Reads the LeBeau Group pymultislicer file at filename, returning a DataCube.

    pymultislicer raw files are shaped as 130x128 arrays, consisting of 128x128 arrays of data followed by
    two rows of metadata.  For each frame, its position in the scan is embedded in the metadata.
    By extracting the scan position of the first and last frames, the function determines the scan
    size. Then, the full dataset is loaded and cropped to the 128x128 valid region.

    Accepts:
        filename    (str) path to the EMPAD file

    Returns:
        data        (DataCube) the 4D datacube, excluding the metadata rows.
    """
    assert (isinstance(filename, (str, Path))), "Error: filepath fp must be a string or pathlib.Path"
    assert (mem in ['RAM', 'MEMMAP']), 'Error: argument mem must be either "RAM" or "MEMMAP"'
    assert (isinstance(binfactor, int)), "Error: argument binfactor must be an integer"
    assert (binfactor >= 1), "Error: binfactor must be >= 1"
    assert (metadata is False), "Error: EMPAD Reader does not support metadata."

    k1 = 130
    k2 = 128
    fPath = Path(filename)

    fnm_parts = filename.split('_')
    print(fnm_parts)
    # Get the scan shape
    shape0 = fnm_parts[-1][1:-4]
    print(shape0)
    shape1 = fnm_parts[-2][1:]
    rShape = 1 + int(shape1[0:2]) - int(shape0[0:2]) # scan shape
    data_shape = (int(shape0), int(shape1), k1, k2)

    # Load the data
    if (mem, binfactor) == ("RAM", 1):
        with open(fPath, "rb") as fid:
            data = np.fromfile(fid, np.float32).reshape(data_shape)[:, :, :128, :]

    elif (mem, binfactor) == ("MEMMAP", 1):
        data = np.memmap(fPath, dtype=np.float32, mode="r", shape=data_shape)[
               :, :, :128, :
               ]

    elif (mem) == ("RAM"):
        # binned read into RAM
        memmap = np.memmap(fPath, dtype=np.float32, mode="r", shape=data_shape)[
                 :, :, :128, :
                 ]
        Q_Nx, Q_Ny, R_Nx, R_Ny = memmap.shape
        Q_Nx, Q_Ny = Q_Nx // binfactor, Q_Ny // binfactor
        data = np.zeros((Q_Nx, Q_Ny, R_Nx, R_Ny), dtype=np.float32)
        for Rx, Ry in tqdmnd(
                R_Nx, R_Ny, desc="Binning data", unit="DP", unit_scale=True
        ):
            data[:, :, Rx, Ry] = bin2D(
                memmap[:, :, Rx, Ry], binfactor, dtype=np.float32
            )

    else:
        # memory mapping + bin-on-load is not supported
        raise Exception(
            "Memory mapping and on-load binning together is not supported.  Either set binfactor=1 or mem='RAM'."
        )
        return

    # data = np.swapaxes(data, 0, 1)
    # data = np.swapaxes(data, 2, 3)
    return DataCube(data=data)


def save_pymultislicer(cube, filename):
    deadrows = np.zeros((2, 128)).astype(np.float32)
    with open(filename, 'wb') as f:

        for px_x in range(cube.R_Nx):
            for px_y in range(cube.R_Ny):
                f.write(bytearray(cube.data[px_x, px_y, :, :]))
                f.write(bytearray(deadrows))