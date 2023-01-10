from num2words import num2words

from . import __version__


def nb_runs_str(nb_runs: int):
    if nb_runs == 1:
        return f"{num2words(nb_runs).title()} run"
    else:
        return f"{num2words(nb_runs).title()} runs"


def general_acquisition_info(desc_data):
    return f"""MR data were acquired using a {desc_data["tesla"]}-Tesla
    {desc_data["manufacturer"]} {desc_data["model"]}."""


def footer():
    return f"This section was (in part) generated automatically using pybids {__version__}."


def _mri_info(desc_data):
    return f"""repetition time, TR={desc_data["tr"]}ms;
    flip angle, FA={desc_data["flip_angle"]}<deg>;
field of view, FOV={desc_data["fov"]}mm;
matrix size={desc_data["matrix_size"]};
voxel size={desc_data["voxel_size"]}mm;"""


def func_info(desc_data):
    return f"""{nb_runs_str(desc_data["nb_runs"])} of {desc_data["task_name"]}
    {desc_data["variants"]}
{desc_data["seqs"]} {desc_data["multi_echo"]} fMRI data were collected
({_mri_info(desc_data)} echo time, TE={desc_data["echo_time"]}ms;
{desc_data["slice_order"]};
{desc_data["multiband_factor"]}; {desc_data["inplaneaccel_str"]}).
Run duration was {desc_data["duration"]} minutes, during which
{desc_data["nb_vols"]} volumes were acquired."""


def anat_info(desc_data):
    return f"""{nb_runs_str(desc_data["nb_runs"])} of {desc_data["scan_type"]}
    {desc_data["variants"]}
{desc_data["seqs"]} {desc_data["multi_echo"]} structural MRI data were collected
({_mri_info(desc_data)} echo time, TE={desc_data["echo_time"]}ms; {desc_data["slice_order"]};
{desc_data["echo_time"]})."""


def dwi_info(desc_data):
    return f"""{nb_runs_str(desc_data["nb_runs"])} of {desc_data["variants"]}
{desc_data["seqs"]} diffusion-weighted (dMRI) data were collected ({_mri_info(desc_data)}
echo time, TE={desc_data["echo_time"]}ms;
b-values of {desc_data["bvals"]}acquired; {desc_data["dmri_dir"]} diffusion directions;
{desc_data["multiband_factor"]})."""


def fmap_info(desc_data):
    return f"""A {desc_data["variants"]} {desc_data["seqs"]} fieldmap ({_mri_info(desc_data)}
phase encoding: {desc_data["multiband_factor"]};
{desc_data["nb_slices"]} slices {desc_data["slice_order"]};
echo time 1 / 2, TE1/2={desc_data["te_1"]}/{desc_data["te_1"]};)
was acquired for the {desc_data["intended_for"]}.
"""
