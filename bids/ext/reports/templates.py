from __future__ import annotations

from pathlib import Path
from typing import Any

import chevron


def render(template_name: str, data: dict[str, Any] | None = None) -> str:
    template_file = Path(__file__).resolve().parent / "templates" / "templates" / template_name

    with open(template_file) as template:
        args = {
            "template": template,
            "data": data,
            "partials_path": Path(__file__).parent.joinpath("templates", "partials"),
            #  keep is only available if chevron is installed from github
            # "keep": True,
        }
        return chevron.render(**args)


def footer() -> str:
    # Imported here to avoid a circular import
    from . import __version__

    return f"This section was (in part) generated automatically using pybids {__version__}."


def anat_info(desc_data: dict[str, Any]) -> str:
    return render(template_name="anat.mustache", data=desc_data)


def func_info(desc_data: dict[str, Any]) -> str:
    return render(template_name="func.mustache", data=desc_data)


def dwi_info(desc_data: dict[str, Any]) -> str:
    return render(template_name="dwi.mustache", data=desc_data)


def fmap_info(desc_data: dict[str, Any]) -> str:
    return render(template_name="fmap.mustache", data=desc_data)


def pet_info(desc_data: dict[str, Any]) -> str:
    return render(template_name="pet.mustache", data=desc_data)


def meg_info(desc_data: dict[str, Any]) -> str:
    return render(template_name="meeg.mustache", data=desc_data)
