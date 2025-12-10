import pytest

from eir.setup.config_validation import validate_no_duplicate_columns


def test_validate_no_duplicate_columns_no_duplicates():
    columns = ["col1", "col2", "col3"]
    validate_no_duplicate_columns(
        columns=columns,
        config_type="Tabular input",
        config_name="test_input",
    )


def test_validate_no_duplicate_columns_with_duplicates():
    columns = ["col1", "col2", "col1", "col3"]
    with pytest.raises(ValueError, match="duplicate column names"):
        validate_no_duplicate_columns(
            columns=columns,
            config_type="Tabular input",
            config_name="test_input",
        )


def test_validate_no_duplicate_columns_multiple_duplicates():
    columns = ["col1", "col2", "col1", "col3", "col2"]
    with pytest.raises(ValueError, match="duplicate column names"):
        validate_no_duplicate_columns(
            columns=columns,
            config_type="Tabular output",
            config_name="test_output",
        )


def test_validate_no_duplicate_columns_case_sensitive():
    columns = ["HDL Cholesterol", "HDL cholesterol"]
    with pytest.raises(ValueError, match="duplicate column names") as exc_info:
        validate_no_duplicate_columns(
            columns=columns,
            config_type="Tabular output",
            config_name="test_output",
        )
    assert "HDL cholesterol" in str(exc_info.value)


def test_validate_no_duplicate_columns_empty_list():
    columns = []
    validate_no_duplicate_columns(
        columns=columns,
        config_type="Tabular input",
        config_name="test_input",
    )


def test_validate_no_duplicate_columns_single_column():
    columns = ["col1"]
    validate_no_duplicate_columns(
        columns=columns,
        config_type="Tabular input",
        config_name="test_input",
    )
