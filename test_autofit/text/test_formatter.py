import os
import shutil
from pathlib import Path

from autofit.text import formatter as frm

text_path = Path(__file__).resolve().parent / "files" / "text"


def test__value_result_string():
    str0 = frm.value_result_string_from(parameter_name="param0", value=2.0)
    assert str0 == "2.00"

    str0 = frm.value_result_string_from(parameter_name="param11", value=3.00)

    assert str0 == "3.0000"

    str0 = frm.value_result_string_from(
        parameter_name="param0", value=2.0, values_at_sigma=(1.5, 2.5)
    )
    assert str0 == "2.00 (1.50, 2.50)"

    str0 = frm.value_result_string_from(
        parameter_name="param0", value=2.0, unit="arcsec"
    )

    assert str0 == "2.00 arcsec"


def test__parameter_result_latex():
    str0 = frm.parameter_result_latex_from(parameter_name="param0", value=2.0)
    assert str0 == r"param0 = 2.00 & "

    str0 = frm.parameter_result_latex_from(
        parameter_name="param0", value=2.0, errors=(0.1, 0.2)
    )
    assert str0 == r"param0 = 2.00^{+0.20}_{-0.10} & "

    str0 = frm.parameter_result_latex_from(
        parameter_name="param0", value=3.00, superscript="a", errors=(0.1, 0.2)
    )

    assert str0 == r"param0^{\rm{a}} = 3.00^{+0.20}_{-0.10} & "

    str0 = frm.parameter_result_latex_from(
        parameter_name="param0", value=3.00, superscript="a", name_to_label=True
    )

    assert str0 == r"p0^{\rm{a}} = 3.00 & "

    str0 = frm.parameter_result_latex_from(
        parameter_name="param0", value=3.00, superscript="a", unit="kg"
    )

    assert str0 == r"param0^{\rm{a}} = 3.00 kg & "


def test__output_list_of_strings_to_file():
    if text_path.exists():
        shutil.rmtree(text_path)

    os.mkdir(text_path)

    results = ["hi\n", "hello"]
    frm.output_list_of_strings_to_file(
        file=text_path / "model.results", list_of_strings=results
    )

    file = open(text_path / "model.results", "r")

    assert file.readlines() == ["hi\n", "hello"]


def test_string():
    assert frm.format_string_for_parameter_name("radius") == "{:.2f}"
    assert frm.format_string_for_parameter_name("mass") == "{:.2f}"


def test_substring():
    assert frm.format_string_for_parameter_name("einstein_radius") == "{:.2f}"
    assert frm.format_string_for_parameter_name("mass_value_something") == "{:.2f}"
