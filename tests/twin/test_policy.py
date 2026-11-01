from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def test_twin_policy_vocabularies_match_the_published_product_boundary():
    from fluidmix.core import policy as product
    from twin import policy as twin

    assert twin.FORBIDDEN_DEPLOY_INPUT_FIELDS == product.FORBIDDEN_DEPLOY_INPUT_FIELDS
    assert twin.DEPLOY_INPUT_ALLOWED_FIELDS == product.DEPLOY_INPUT_ALLOWED_FIELDS


@pytest.mark.parametrize(
    ("value", "require_allowlist"),
    [
        (
            {
                "case_id": "CASE-1",
                "recipe_ml": {"t100": 10, "t200": 20, "t400": 0},
                "source_capture_manifest_sha256": {"CASE-1": "a" * 64},
            },
            True,
        ),
        ({"published_schema_field": {"nested": "value"}}, False),
    ],
)
def test_twin_policy_acceptance_matches_product_policy(value, require_allowlist):
    from fluidmix.core.policy import validate_deploy_input_fields as product_validate
    from twin.policy import validate_deploy_input_fields as twin_validate

    assert product_validate(value, require_allowlist=require_allowlist) is None
    assert twin_validate(value, require_allowlist=require_allowlist) is None


@pytest.mark.parametrize("field", ["oracle", "label", "composition"])
def test_twin_policy_rejects_the_same_forbidden_fields_as_product(field):
    from fluidmix.core.policy import validate_deploy_input_fields as product_validate
    from twin.policy import validate_deploy_input_fields as twin_validate

    value = {"case_id": "CASE-1", field: "hidden"}
    for validate in (product_validate, twin_validate):
        with pytest.raises(ValueError, match=f"forbidden_deploy_input_field: {field}"):
            validate(value, require_allowlist=False)


def test_twin_policy_rejects_unknown_allowlist_field_like_product():
    from fluidmix.core.policy import validate_deploy_input_fields as product_validate
    from twin.policy import validate_deploy_input_fields as twin_validate

    for validate in (product_validate, twin_validate):
        with pytest.raises(
            ValueError, match="forbidden_deploy_input_field: arbitrary"
        ):
            validate({"arbitrary": "value"})
