from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from nibabel import Nifti1Image
from num2words import num2words

from .logger import pybids_reports_logger
from .utils import list_to_str
from .utils import num_to_str
from .utils import remove_duplicates
from bids.layout import BIDSFile
from bids.layout import BIDSLayout

"""Functions for building strings for individual parameters."""
LOGGER = pybids_reports_logger()


def nb_runs(run_list: list[str]) -> str:
    nb_runs = len(run_list)
    if nb_runs == 1:
        return f"{num2words(nb_runs).title()} run"
    else:
        return f"{num2words(nb_runs).title()} runs"


def slice_order(metadata: dict[str, Any]) -> str:
    """Generate description of slice timing from metadata."""
    if "SliceTiming" in metadata:
        return f' in {get_slice_info(metadata["SliceTiming"])} order'
    else:
        return ""


def func_duration(nb_vols: int, tr: float) -> str:
    """Generate description of functional run length from repetition time and number of volumes."""
    run_secs = math.ceil(nb_vols * tr)
    mins, secs = divmod(run_secs, 60)
    return f"{mins}:{secs}"


def get_nb_vols(all_imgs: list[Nifti1Image | None]) -> list[int] | None:
    """Get number of volumes from list of files.

    If all files have the same nb of vols it will return the number of volumes,
    otherwise it will return the minimum and maximum number of volumes.
    """
    nb_vols = []
    for img in all_imgs:
        if img is not None:
            nb_vols.append(img.shape[3])
        else:
            nb_vols.append(None)
    nb_vols = list(set(nb_vols))

    if nb_vols == [None]:
        return None

    if len(nb_vols) <= 1:
        return [nb_vols[0]]

    min_vols = min(nb_vols)
    max_vols = max(nb_vols)
    return [min_vols, max_vols]


def nb_vols(all_imgs: list[Nifti1Image]) -> str:
    nb_vols = get_nb_vols(all_imgs)
    if nb_vols is None:
        return "UNKNOWN"
    return f"{nb_vols[0]}-{nb_vols[1]}" if len(nb_vols) > 1 else str(nb_vols[0])


def duration(all_imgs: list[Nifti1Image], metadata: dict[str, Any]) -> str:
    """Generate general description of scan length from files."""

    nb_vols = get_nb_vols(all_imgs)
    if nb_vols is None:
        return "UNKNOWN"
    tr = metadata["RepetitionTime"]
    if len(nb_vols) <= 1:
        return func_duration(nb_vols[0], tr)

    min_dur = func_duration(nb_vols[0], tr)
    max_dur = func_duration(nb_vols[1], tr)
    return f"{min_dur}-{max_dur}"


def multiband_factor(metadata: dict[str, Any]) -> str:
    """Generate description of the multi-band acceleration applied, if used."""
    return (
        f'MB factor={metadata["MultibandAccelerationFactor"]}'
        if metadata.get("MultibandAccelerationFactor", 1) > 1
        else ""
    )


def echo_time_ms(files: list[BIDSFile]) -> str:
    """Generate description of echo times from metadata field.

    Parameters
    ----------
    files : :obj:`list` of :obj:`bids.layout.models.BIDSFile`
        List of nifti files in layout corresponding to file collection.

    Returns
    -------
    te : str
        Description of echo times.
    """
    echo_times = [f.get_metadata().get("EchoTime", None) for f in files]
    echo_times = sorted(list(set(echo_times)))
    if echo_times == [None]:
        return "UNKNOWN"
    if len(echo_times) <= 1:
        return num_to_str(echo_times[0] * 1000)
    te = [num_to_str(t * 1000) for t in echo_times]
    return list_to_str(te)


def multi_echo(files: list[BIDSFile]) -> str:
    """Generate description of echo times from metadata field.

    Parameters
    ----------
    files : :obj:`list` of :obj:`bids.layout.models.BIDSFile`
        List of nifti files in layout corresponding to file collection.

    Returns
    -------
    multi_echo : str
        Whether the data are multi-echo or single-echo.
    """

    echo_times = [f.get_metadata().get("EchoTime", None) for f in files]
    echo_times = sorted(list(set(echo_times)))
    if echo_times == [None]:
        return ""
    multi_echo = "multi-echo" if len(echo_times) > 1 else "single-echo"
    return multi_echo


def echo_times_fmap(files: list[BIDSFile]) -> tuple[float, float]:
    """Generate description of echo times from metadata field for fmaps

    Parameters
    ----------
    files : :obj:`list` of :obj:`bids.layout.models.BIDSFile`
        List of nifti files in layout corresponding to file collection.

    Returns
    -------
    te_str :
        Description of echo times.
    """
    # TODO handle all types of fieldmaps

    echo_times1 = [f.get_metadata()["EchoTime1"] for f in files]
    echo_times2 = [f.get_metadata()["EchoTime2"] for f in files]
    echo_times1 = sorted(list(set(echo_times1)))
    echo_times2 = sorted(list(set(echo_times2)))
    if len(echo_times1) <= 1 and len(echo_times2) <= 1:
        # if that's not the case we should probably throw a warning
        # because we should expect the same echo times for all values
        te1 = echo_times1[0] * 1000
        te2 = echo_times2[0] * 1000
    return te1, te2


def inplane_accel(metadata: dict[str, Any]) -> str:
    """Generate description of in-plane acceleration factor, if any."""
    return (
        f'in-plane acceleration factor={metadata["ParallelReductionFactorInPlane"]}'
        if metadata.get("ParallelReductionFactorInPlane", 1) > 1
        else ""
    )


def bvals(bval_file: str | Path) -> str:
    """Generate description of dMRI b-values."""
    # Parse bval file
    with open(bval_file) as file_object:
        raw_bvals = file_object.read().splitlines()
    # Flatten list of space-separated values
    bvals = [item for sublist in [line.split(" ") for line in raw_bvals] for item in sublist]
    bvals_as_int = sorted([int(v) for v in set(bvals) if v not in [""]])
    bvals_as_list = [num_to_str(v) for v in bvals_as_int]
    return list_to_str(bvals_as_list)


def intendedfor_targets(metadata: dict[str, Any], layout: BIDSLayout) -> str:
    """Generate description of intended for targets."""

    if "IntendedFor" not in metadata:
        return ""

    scans = metadata["IntendedFor"]

    tmp_dict: dict[str, list[int]] = {}

    for scan in scans:
        fn = Path(scan).name

        if_file = [f for f in layout.get(extension=[".nii", ".nii.gz"]) if fn in f.path][0]

        target_type = if_file.entities["suffix"].upper()
        if target_type == "BOLD":
            iff_meta = layout.get_metadata(if_file.path)
            task = iff_meta.get("TaskName", if_file.entities["task"])
            target_type_str = f"{task} {target_type} scan"
        else:
            target_type_str = f"{target_type} scan"

        run_num = int(if_file.run)
        if target_type_str in tmp_dict:
            tmp_dict[target_type_str].append(run_num)
        else:
            tmp_dict[target_type_str] = [run_num]

    run_dict: dict[str, list[str]] = {
        scan: [num2words(r, ordinal=True) for r in sorted(tmp_dict[scan])] for scan in tmp_dict
    }

    out_list = []
    for scan in run_dict:
        s = "s" if len(run_dict[scan]) > 1 else ""
        run_str = list_to_str(run_dict[scan])
        string = f"{run_str} run{s} of the {scan}"
        out_list.append(string)

    return list_to_str(out_list)


def get_slice_info(slice_times: list[Any]) -> str | list[str]:
    """Extract slice order from slice timing info.

    TODO: Be more specific with slice orders.
    Currently anything where there's some kind of skipping is interpreted as
    interleaved of some kind.

    Parameters
    ----------
    slice_times : array-like
        A list of slice times in seconds or milliseconds or whatever.

    Returns
    -------
    slice_order_name : :obj:`str`
        The name of the slice order sequence.
    """
    # Slice order
    slice_times = remove_duplicates(slice_times)
    slice_order = sorted(range(len(slice_times)), key=lambda k: slice_times[k])

    if slice_order == list(range(len(slice_order))):
        slice_order_name = "sequential ascending"

    elif slice_order == list(reversed(range(len(slice_order)))):
        slice_order_name = "sequential descending"

    elif slice_order[0] < slice_order[1]:
        # We're allowing some wiggle room on interleaved.
        slice_order_name = "interleaved ascending"

    elif slice_order[0] > slice_order[1]:
        slice_order_name = "interleaved descending"

    else:
        raise Exception(f"Unknown slice order: [{', '.join([str(s) for s in slice_order])}]")

    return slice_order_name


def variants(metadata: dict[str, Any], config: dict[str, dict[str, str]]) -> str:
    """Extract and reformat imaging variant(s).

    Parameters
    ----------
    metadata : :obj:`dict`
        The metadata for the scan.

    config : :obj:`dict`
        A dictionary with relevant information regarding sequences, sequence
        variants, phase encoding directions, and task names.

    Returns
    -------
    variants : :obj:`str`
        Sequence variant names.
    """
    variants = metadata.get("SequenceVariant", "")
    if isinstance(variants, str):
        variants = [
            config["seqvar"].get(var, "UNKNOWN SEQUENCE VARIANT") for var in variants.split("_")
        ]
    return list_to_str(variants)


def sequence(metadata: dict[str, Any], config: dict[str, dict[str, str]]) -> str | list[str]:
    """Extract and reformat imaging sequence(s) and variant(s) into pretty strings.

    Parameters
    ----------
    metadata : :obj:`dict`
        The metadata for the scan.

    config : :obj:`dict`
        A dictionary with relevant information regarding sequences, sequence
        variants, phase encoding directions, and task names.

    Returns
    -------
    seqs : :obj:`str`
        Sequence names.
    """
    seq_abbrs = metadata.get("ScanningSequence", "")
    if isinstance(seq_abbrs, str):
        seq_abbrs = seq_abbrs.split("_")
    seqs = [config["seq"].get(seq, "") for seq in seq_abbrs]
    seqs_as_str = list_to_str(seqs)
    if seq_abbrs[0] and seqs_as_str:
        seqs_as_str += f" ({'/'.join(seq_abbrs)})"
    else:
        seqs_as_str = "UNKNOWN SEQUENCE"

    return seqs_as_str


def matrix_size(img: None | Nifti1Image) -> str:
    """Extract and reformat voxel size, matrix size, FOV, and number of slices into strings.

    Parameters
    ----------
    img : :obj:`nibabel.Nifti1Image`
        Image from scan from which to derive parameters.

    Returns
    -------
    matrix_size : :obj:`str`
        Matrix size string (e.g., '128x128')
    """
    if img is None:
        return "?x?"
    n_x, n_y = img.shape[:2]
    return f"{n_x}x{n_y}"


def voxel_size(img: None | Nifti1Image) -> str:
    """Extract and reformat voxel size.

    Parameters
    ----------
    img : :obj:`nibabel.Nifti1Image`
        Image from scan from which to derive parameters.

    Returns
    -------
    voxel_size : :obj:`str`
        Voxel size string (e.g., '2x2x2')
    """
    if img is None:
        return "?x?x?"
    voxel_dims = np.array(img.header.get_zooms()[:3])
    return "x".join([num_to_str(s) for s in voxel_dims])


def field_of_view(img: None | Nifti1Image) -> str:
    """Extract and reformat FOV.

    Parameters
    ----------
    img : :obj:`nibabel.Nifti1Image`
        Image from scan from which to derive parameters.

    Returns
    -------
    fov : :obj:`str`
        Field of view string (e.g., '256x256')
    """
    if img is None:
        return "?x?"
    n_x, n_y = img.shape[:2]
    voxel_dims = np.array(img.header.get_zooms()[:3])
    fov = [n_x, n_y] * voxel_dims[:2]
    return "x".join([num_to_str(s) for s in fov])
