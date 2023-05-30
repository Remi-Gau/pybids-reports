"""Parsing functions for generating BIDSReports."""
import logging
import warnings

import nibabel as nib

from . import parameters, templates
from .utils import collect_associated_files

logging.basicConfig()
LOGGER = logging.getLogger("pybids-reports.parsing")


def common_mri_desc(img, metadata: dict, config: dict) -> dict:
    return {
        **metadata,
        "field_strength": metadata.get("MagneticFieldStrength", "UNKNOWN"),
        "tr": metadata["RepetitionTime"] * 1000,
        "flip_angle": metadata.get("FlipAngle", "UNKNOWN"),
        "fov": parameters.field_of_view(img),
        "matrix_size": parameters.matrix_size(img),
        "voxel_size": parameters.voxel_size(img),
        "variants": parameters.variants(metadata, config),
        "seqs": parameters.sequence(metadata, config),
        "nb_slices": len(metadata["SliceTiming"]) if "SliceTiming" in metadata else img.shape[2],
    }


def func_info(files, config):
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
    first_file = files[0]
    metadata = first_file.get_metadata()
    img = nib.load(first_file.path)
    all_imgs = [nib.load(f) for f in files]

    task_name = first_file.get_entities()["task"]

    all_runs = sorted(list({f.get_entities().get("run", 1) for f in files}))

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
        "nb_vols": parameters.nb_vols(all_imgs),
        "duration": parameters.duration(all_imgs, metadata),
        "scan_type": first_file.get_entities()["suffix"].replace("w", "-weighted"),
    }

    return templates.func_info(desc_data)


def anat_info(files, config):
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
    img = nib.load(first_file.path)

    all_runs = sorted(list({f.get_entities().get("run", 1) for f in files}))

    desc_data = {
        **common_mri_desc(img, metadata, config),
        "echo_time": parameters.echo_time_ms(files),
        "slice_order": parameters.slice_order(metadata),
        "nb_runs": parameters.nb_runs(all_runs),
        "multi_echo": parameters.multi_echo(files),
    }

    return templates.anat_info(desc_data)


def dwi_info(files, config):
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
    img = nib.load(first_file.path)
    bval_file = first_file.path.replace(".nii.gz", ".bval").replace(".nii", ".bval")

    all_runs = sorted(list({f.get_entities().get("run", 1) for f in files}))

    desc_data = {
        **common_mri_desc(img, metadata, config),
        "echo_time": parameters.echo_time_ms(files),
        "nb_runs": parameters.nb_runs(all_runs),
        "bvals": parameters.bvals(bval_file),
        "dmri_dir": img.shape[3],
        "multiband_factor": parameters.multiband_factor(metadata),
    }

    return templates.dwi_info(desc_data)


def fmap_info(layout, files, config):
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
    img = nib.load(first_file.path)

    desc_data = {
        **common_mri_desc(img, metadata, config),
        "te_1": parameters.echo_times_fmap(files)[0],
        "te_2": parameters.echo_times_fmap(files)[1],
        "slice_order": parameters.slice_order(metadata),
        "dir": config["dir"].get(metadata["PhaseEncodingDirection"], "UNKNOWN PHASE ENCODING"),
        "multiband_factor": parameters.multiband_factor(metadata),
        "intended_for": parameters.intendedfor_targets(metadata, layout),
    }

    return templates.fmap_info(desc_data)


def meg_info(files):
    """Generate a paragraph describing meg acquisition information.

    Parameters
    ----------
    layout : :obj:`bids.layout.BIDSLayout`
        Layout object for a BIDS dataset.
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


def device_info(metadata):
    return {
        "manufacturer": metadata.get("Manufacturer", "MANUFACTURER"),
        "model_name": metadata.get("ManufacturersModelName", "MODEL"),
    }


def final_paragraph(metadata):
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
    if "ConversionSoftware" in metadata.keys():
        soft = metadata["ConversionSoftware"]
        vers = metadata["ConversionSoftwareVersion"]
        software_str = f" using {soft} ({vers})"
    else:
        software_str = ""
    return f"Dicoms were converted to NIfTI-1 format{software_str}."


def parse_files(layout, data_files, config):
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

    # print(data_files)

    # description_list = [general_acquisition_info(data_files[0][0].get_metadata())]
    description_list = []
    for group in data_files:
        if group[0].entities["datatype"] == "func":
            group_description = func_info(group, config)

        elif (group[0].entities["datatype"] == "anat") and group[0].entities["suffix"].endswith(
            "w"
        ):
            group_description = anat_info(group, config)

        elif group[0].entities["datatype"] == "dwi":
            group_description = dwi_info(group, config)

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
            warnings.warn(group[0].entities["datatype"] + " not yet supported.")
            continue

        else:
            warnings.warn(f"{group[0].filename} not yet supported.")
            continue

        description_list.append(group_description)

    return description_list
