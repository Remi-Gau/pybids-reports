"""Parsing functions for generating BIDSReports."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
from nibabel.filebasedimages import ImageFileError

from . import parameters
from . import templates
from .logger import pybids_reports_logger
from .utils import collect_associated_files
from bids.layout import BIDSFile
from bids.layout import BIDSLayout

LOGGER = pybids_reports_logger()


def common_mri_desc(
    img: None | nib.Nifti1Image, metadata: dict[str, Any], config: dict[str, dict[str, str]]
) -> dict[str, Any]:
    nb_slices = "UNKNOWN"
    if "SliceTiming" in metadata:
        nb_slices = str(len(metadata["SliceTiming"]))
    if img is not None and not nb_slices:
        nb_slices = str(img.shape[2])

    tr = "UNKNOWN"
    if "RepetitionTime" in metadata:
        tr = metadata["RepetitionTime"] * 1000

    return {
        **metadata,
        "field_strength": metadata.get("MagneticFieldStrength", "UNKNOWN"),
        "tr": tr,
        "flip_angle": metadata.get("FlipAngle", "UNKNOWN"),
        "fov": parameters.field_of_view(img),
        "matrix_size": parameters.matrix_size(img),
        "voxel_size": parameters.voxel_size(img),
        "variants": parameters.variants(metadata, config),
        "seqs": parameters.sequence(metadata, config),
        "nb_slices": nb_slices,
    }


def func_info(files: list[BIDSFile], config: dict[str, dict[str, str]], layout: BIDSLayout) -> str:
    """Generate a paragraph describing T2*-weighted functional scans.

    Parameters
    ----------
    files : :obj:`list` of :obj:`bids.layout.models.BIDSFile`
        List of nifti files in layout corresponding to DWI scan.

    config : :obj:`dict`
        A dictionary with relevant information regarding sequences, sequence
        variants, phase encoding directions, and task names.

    Returns
    -------
    desc : :obj:`str`
        A description of the scan's acquisition information.
    """
    errored_files = []

    first_file = files[0]
    metadata = first_file.get_metadata()
    img = try_load_nii(first_file.path)
    if img is None:
        errored_files.append(Path(first_file.path).relative_to(layout.root))

    all_imgs = []
    for f in files:
        img = try_load_nii(f)
        if img is None:
            errored_files.append(Path(f).relative_to(layout.root))
        else:
            all_imgs.append(img)
    if errored_files:
        files_not_found_warning(list(set(errored_files)))

    task_name = first_file.get_entities()["task"]

    all_runs = sorted(list({f.get_entities().get("run", 1) for f in files}))

    nb_vols = "UNKNOWN"
    duration = "UNKNOWN"
    if all_imgs:
        nb_vols = parameters.nb_vols(all_imgs)
        duration = parameters.duration(all_imgs, metadata)

    desc_data = {
        **common_mri_desc(img, metadata, config),
        **device_info(metadata),
        "echo_time": parameters.echo_time_ms(files),
        "slice_order": parameters.slice_order(metadata),
        "multiband_factor": parameters.multiband_factor(metadata),
        "inplane_accel": parameters.inplane_accel(metadata),
        "nb_runs": parameters.nb_runs(all_runs),
        "task_name": metadata.get("TaskName", task_name),
        "multi_echo": parameters.multi_echo(files),
        "nb_vols": nb_vols,
        "duration": duration,
        "scan_type": first_file.get_entities()["suffix"].replace("w", "-weighted"),
    }

    return templates.func_info(desc_data)


def anat_info(files: list[BIDSFile], config: dict[str, dict[str, str]], layout: BIDSLayout) -> str:
    """Generate a paragraph describing T1- and T2-weighted structural scans.

    Parameters
    ----------
    files : :obj:`list` of :obj:`bids.layout.models.BIDSFile`
        List of nifti files in layout corresponding to DWI scan.

    config : :obj:`dict`
        A dictionary with relevant information regarding sequences, sequence
        variants, phase encoding directions, and task names.

    Returns
    -------
    desc : :obj:`str`
        A description of the scan's acquisition information.
    """
    first_file = files[0]
    metadata = first_file.get_metadata()
    img = try_load_nii(first_file.path)
    if img is None:
        files_not_found_warning(Path(first_file.path).relative_to(layout.root))

    all_runs = sorted(list({f.get_entities().get("run", 1) for f in files}))

    desc_data = {
        **common_mri_desc(img, metadata, config),
        "echo_time": parameters.echo_time_ms(files),
        "slice_order": parameters.slice_order(metadata),
        "nb_runs": parameters.nb_runs(all_runs),
        "multi_echo": parameters.multi_echo(files),
    }

    return templates.anat_info(desc_data)


def dwi_info(files: list[BIDSFile], config: dict[str, dict[str, str]], layout: BIDSLayout) -> str:
    """Generate a paragraph describing DWI scan acquisition information.

    Parameters
    ----------
    files : :obj:`list` of :obj:`bids.layout.models.BIDSFile`
        List of nifti files in layout corresponding to DWI scan.

    config : :obj:`dict`
        A dictionary with relevant information regarding sequences, sequence
        variants, phase encoding directions, and task names.

    Returns
    -------
    desc : :obj:`str`
        A description of the DWI scan's acquisition information.
    """
    first_file = files[0]
    metadata = first_file.get_metadata()
    img = try_load_nii(first_file.path)
    if img is None:
        files_not_found_warning(Path(first_file.path).relative_to(layout.root))
    bval_file = first_file.path.replace(".nii.gz", ".bval").replace(".nii", ".bval")

    all_runs = sorted(list({f.get_entities().get("run", 1) for f in files}))

    dmri_dir = "UNKNOWN"
    if img is not None:
        dmri_dir = img.shape[3]

    desc_data = {
        **common_mri_desc(img, metadata, config),
        "echo_time": parameters.echo_time_ms(files),
        "nb_runs": parameters.nb_runs(all_runs),
        "bvals": parameters.bvals(bval_file),
        "dmri_dir": dmri_dir,
        "multiband_factor": parameters.multiband_factor(metadata),
    }

    return templates.dwi_info(desc_data)


def fmap_info(layout: BIDSLayout, files: list[BIDSFile], config: dict[str, dict[str, str]]) -> str:
    """Generate a paragraph describing field map acquisition information.

    Parameters
    ----------
    layout : :obj:`bids.layout.BIDSLayout`
        Layout object for a BIDS dataset.

    files : :obj:`list` of :obj:`bids.layout.models.BIDSFile`
        List of nifti files in layout corresponding to field map scan.

    config : :obj:`dict`
        A dictionary with relevant information regarding sequences, sequence
        variants, phase encoding directions, and task names.

    Returns
    -------
    desc : :obj:`str`
        A description of the field map's acquisition information.
    """
    first_file = files[0]
    metadata = first_file.get_metadata()
    img = try_load_nii(first_file.path)
    if img is None:
        files_not_found_warning(Path(first_file.path).relative_to(layout.root))

    direction = "UNKNOWN PHASE ENCODING"
    if PhaseEncodingDirection := metadata.get("PhaseEncodingDirection"):
        direction = config["dir"].get(PhaseEncodingDirection, "UNKNOWN PHASE ENCODING")

    desc_data = {
        **common_mri_desc(img, metadata, config),
        "te_1": parameters.echo_times_fmap(files)[0],
        "te_2": parameters.echo_times_fmap(files)[1],
        "slice_order": parameters.slice_order(metadata),
        "dir": direction,
        "multiband_factor": parameters.multiband_factor(metadata),
        "intended_for": parameters.intendedfor_targets(metadata, layout),
    }

    return templates.fmap_info(desc_data)


def meg_info(files: list[BIDSFile]) -> str:
    """Generate a paragraph describing meg acquisition information.

    Parameters
    ----------
    files : :obj:`list` of :obj:`bids.layout.models.BIDSFile`
        List of nifti files in layout corresponding to meg scan.

    config : :obj:`dict`
        A dictionary with relevant information regarding sequences, sequence
        variants, phase encoding directions, and task names.

    Returns
    -------
    desc : :obj:`str`
        A description of the field map's acquisition information.
    """
    first_file = files[0]
    metadata = first_file.get_metadata()

    desc_data = {**device_info(metadata), **metadata}

    return templates.meg_info(desc_data)


def device_info(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "manufacturer": metadata.get("Manufacturer", "MANUFACTURER"),
        "model_name": metadata.get("ManufacturersModelName", "MODEL"),
    }


def final_paragraph(metadata: dict[str, Any]) -> str:
    """Describe dicom-to-nifti conversion process and methods generation.

    Parameters
    ----------
    metadata : :obj:`dict`
        The metadata for the scan.

    Returns
    -------
    desc : :obj:`str`
        Output string with scanner information.
    """
    if "ConversionSoftware" in metadata:
        soft = metadata["ConversionSoftware"]
        vers = metadata["ConversionSoftwareVersion"]
        software_str = f" using {soft} ({vers})"
    else:
        software_str = ""
    return f"Dicoms were converted to NIfTI-1 format{software_str}."


def parse_files(
    layout: BIDSLayout, data_files: list[BIDSFile], config: dict[str, dict[str, str]]
) -> list[str]:
    """Loop through files in a BIDSLayout and generate appropriate descriptions.

    Then, compile all of the descriptions into a list.

    Parameters
    ----------
    layout : :obj:`bids.layout.BIDSLayout`
        Layout object for a BIDS dataset.

    data_files : :obj:`list` of :obj:`bids.layout.models.BIDSFile`
        List of nifti files in layout corresponding to subject/session combo.

    config : :obj:`dict`
        Configuration info for methods generation.
    """
    # Group files into individual runs
    data_files = collect_associated_files(layout, data_files, extra_entities=["run"])

    description_list = []
    for group in data_files:
        if group[0].entities["datatype"] == "func":
            group_description = func_info(group, config, layout)

        elif (group[0].entities["datatype"] == "anat") and group[0].entities["suffix"] in (
            "T1w",
            "T2w",
            "PDw",
            "T2starw",
            "FLAIR",
            "inplaneT1",
            "inplaneT2",
            "PDT2",
            "angio",
        ):
            group_description = anat_info(group, config, layout)

        elif group[0].entities["datatype"] == "dwi":
            group_description = dwi_info(group, config, layout)

        elif (group[0].entities["datatype"] == "fmap") and group[0].entities[
            "suffix"
        ] == "phasediff":
            group_description = fmap_info(layout, group, config)

        elif group[0].entities["datatype"] in [
            "eeg",
            "meg",
            "pet",
            "ieeg",
            "beh",
            "perf",
            "fnirs",
            "microscopy",
        ]:
            LOGGER.warning(f" '{group[0].entities['datatype']}' not yet supported.")
            group_description = ""

        else:
            LOGGER.warning(f" '{group[0].filename}' not yet supported.")
            group_description = ""

        if group_description:
            description_list.append(group_description)

    return description_list


def try_load_nii(file: BIDSFile) -> None | nib.Nifti1Image:
    try:
        img = nib.load(file)
    except (FileNotFoundError, ImageFileError):
        img = None
    return img


def files_not_found_warning(files: list[BIDSFile] | BIDSFile) -> None:
    if not isinstance(files, list):
        files = [files]
    files = [str(Path(file)) for file in files]
    LOGGER.warning(f"File not found or empty:\n {files}")
