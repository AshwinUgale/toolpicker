"""Tests for OpenAPISource."""

from __future__ import annotations

from typing import Any

import pytest

from toolpicker import OpenAPISource, ToolPicker

# A Petstore-flavored spec covering: operationId present + missing, path
# params, query params, JSON request body, $ref resolution.
_PETSTORE: dict[str, Any] = {
    "openapi": "3.0.0",
    "info": {"title": "Petstore", "version": "1.0.0"},
    "components": {
        "schemas": {
            "NewPet": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string", "description": "Pet name."},
                    "tag": {"type": "string"},
                },
            }
        }
    },
    "paths": {
        "/pets": {
            "get": {
                "operationId": "listPets",
                "summary": "List all pets",
                "description": "Returns up to `limit` pets.",
                "parameters": [
                    {
                        "name": "limit",
                        "in": "query",
                        "schema": {"type": "integer"},
                        "description": "Maximum number of pets to return.",
                    }
                ],
                "tags": ["pets", "read"],
                "responses": {"200": {"description": "A list of pets."}},
            },
            "post": {
                "operationId": "createPet",
                "summary": "Create a pet",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/NewPet"}}
                    },
                },
                "tags": ["pets", "write"],
                "responses": {"201": {"description": "Created."}},
            },
        },
        "/pets/{petId}": {
            "get": {
                # NO operationId on purpose - test the fallback.
                "summary": "Get a pet by id",
                "parameters": [
                    {
                        "name": "petId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                    }
                ],
                "responses": {"200": {"description": "A pet."}},
            }
        },
    },
}


def test_parses_petstore_correctly() -> None:
    source = OpenAPISource(_PETSTORE)
    tools = source.tools()
    by_id = {t.id: t for t in tools}

    # Three operations -> three tools.
    assert set(by_id.keys()) == {"listPets", "createPet", "get_pets_petId"}

    list_tool = by_id["listPets"]
    assert "List all pets" in list_tool.description
    assert "Returns up to" in list_tool.description
    assert "limit" in list_tool.parameters_schema["properties"]
    assert "limit" not in list_tool.parameters_schema.get("required", [])
    assert list_tool.keywords == ["pets", "read"]


def test_path_param_is_required() -> None:
    source = OpenAPISource(_PETSTORE)
    by_id = {t.id: t for t in source.tools()}
    get_pet = by_id["get_pets_petId"]
    assert "petId" in get_pet.parameters_schema["properties"]
    assert get_pet.parameters_schema["required"] == ["petId"]


def test_request_body_object_properties_get_flattened() -> None:
    source = OpenAPISource(_PETSTORE)
    by_id = {t.id: t for t in source.tools()}
    create = by_id["createPet"]
    # The body schema's properties end up at the top level.
    assert "name" in create.parameters_schema["properties"]
    assert "tag" in create.parameters_schema["properties"]
    # Required list comes from the body schema.
    assert "name" in create.parameters_schema["required"]


def test_ref_resolution() -> None:
    # The "createPet" body uses $ref into components/schemas/NewPet. After
    # resolution we should see actual property dicts, not a $ref entry.
    source = OpenAPISource(_PETSTORE)
    by_id = {t.id: t for t in source.tools()}
    create = by_id["createPet"]
    name_schema = create.parameters_schema["properties"]["name"]
    assert name_schema.get("type") == "string"
    assert name_schema.get("description") == "Pet name."


def test_operation_id_fallback() -> None:
    # The path declares {bar}, so each operation must list it as a path
    # parameter (the OpenAPI validator catches unresolved path params).
    path_param = {
        "name": "bar",
        "in": "path",
        "required": True,
        "schema": {"type": "string"},
    }
    spec: dict[str, Any] = {
        "openapi": "3.0.0",
        "info": {"title": "x", "version": "0"},
        "paths": {
            "/foo/{bar}/baz": {
                "get": {
                    "parameters": [path_param],
                    "responses": {"200": {"description": "ok"}},
                },
                "post": {
                    "parameters": [path_param],
                    "responses": {"200": {"description": "ok"}},
                },
            }
        },
    }
    source = OpenAPISource(spec)
    ids = {t.id for t in source.tools()}
    assert ids == {"get_foo_bar_baz", "post_foo_bar_baz"}


def test_validation_can_be_disabled() -> None:
    # Spec missing required `info.version` would normally fail validation;
    # with validate=False, the parse still extracts what's there.
    minimal: dict[str, Any] = {
        "openapi": "3.0.0",
        "info": {"title": "x"},  # no version - invalid per OpenAPI 3.0
        "paths": {
            "/health": {
                "get": {"operationId": "health", "responses": {"200": {"description": "ok"}}}
            }
        },
    }
    source = OpenAPISource(minimal, validate=False)
    assert {t.id for t in source.tools()} == {"health"}


def test_invalid_spec_raises_when_validation_on() -> None:
    invalid: dict[str, Any] = {"openapi": "3.0.0", "info": {"title": "x"}}  # missing paths, version
    with pytest.raises(Exception):  # noqa: B017 - validator may raise various types
        OpenAPISource(invalid, validate=True)


def test_duplicate_operation_ids_raise() -> None:
    # Bypass the spec validator (which catches dupes too) to verify our own
    # post-parse duplicate check fires.
    spec: dict[str, Any] = {
        "openapi": "3.0.0",
        "info": {"title": "x", "version": "0"},
        "paths": {
            "/a": {"get": {"operationId": "dupe", "responses": {"200": {"description": "ok"}}}},
            "/b": {"get": {"operationId": "dupe", "responses": {"200": {"description": "ok"}}}},
        },
    }
    with pytest.raises(ValueError, match="duplicate"):
        OpenAPISource(spec, validate=False)


def test_router_can_consume_openapi_source() -> None:
    """End-to-end: OpenAPISource feeds ToolPicker; lexical query finds the tool."""
    source = OpenAPISource(_PETSTORE)
    picker = ToolPicker(source)
    hits = picker.select("list all the pets in the store", k=3)
    assert any(t.id == "listPets" for t in hits)
