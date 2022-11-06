from FACE.io.Graphics import plot_spn


def overwrite_plot_spn(FACE, plotfile):
    import os
    try:
        os.remove(plotfile)
    except OSError as err:
        pass
    plot_spn(FACE, plotfile)
