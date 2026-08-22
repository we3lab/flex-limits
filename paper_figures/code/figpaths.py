import os


def figure_dir(basepath, ext, si=False):
    """Resolve the output directory for one figure format.

    Manuscript figures go to ``figures/<ext>/``. Supporting-information figures go to
    ``figures/SI/<ext>/``, so that the top level of ``figures/`` holds only the panels that
    appear in the paper and a stale SI render cannot be mistaken for a manuscript figure.

    Parameters
    ----------
    basepath : str
        The ``paper_figures/`` directory, i.e. ``dirname(dirname(__file__))`` from a script
        in ``paper_figures/code/``.
    ext : str
        The image format, and the name of the leaf directory: "png", "pdf" or "svg".
    si : bool, optional
        True for a supporting-information figure. Default is False.

    Returns
    -------
    str
        The directory path, created if it did not already exist.
    """
    parts = [basepath, "figures"] + (["SI"] if si else []) + [ext]
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path
