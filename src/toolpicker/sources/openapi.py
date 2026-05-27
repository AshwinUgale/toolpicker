"""Adapter from OpenAPI 3.0 / 3.1 specs to ``Tool`` objects.

One ``Tool`` per operation. The mapping is mechanical:

* tool ``id`` / ``name`` = ``operationId`` if set, else ``{method}_{path}``
  with path slashes turned into underscores and ``{}`` stripped
  (``GET /pets/{petId}`` -> ``get_pets_petId``)
* tool ``description`` = ``summary`` + ``description`` joined
* tool ``parameters_schema`` = JSON-Schema-shaped object combining the
  operation's ``parameters`` (path / query / header / cookie) and the
  ``requestBody`` JSON schema. Body-object properties are merged into
  the top-level properties dict; non-object bodies nest under ``"body"``.
* tool ``keywords`` = operation ``tags``

``$ref`` resolution uses ``jsonref.replace_refs`` so component schemas are
inlined before we walk operations.

Optional validation via ``openapi-spec-validator`` (on by default). Off
when the user is feeding a known-good spec and wants the import to skip
the validator startup cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from toolpicker.types import Tool

if TYPE_CHECKING:
    pass

__all__ = ["OpenAPISource"]


# OpenAPI's known HTTP methods. We scan paths for these keys only; anything
# else (parameters, summary, x-* extensions) gets ignored at the path level.
_HTTP_METHODS = ("get", "post", "put", "delete", "patch", "head", "options", "trace")


def _path_to_id(path: str) -> str:
    """Convert an OpenAPI path template into a tool-id-safe string.

    ``/pets/{petId}/orders`` -> ``pets_petId_orders``.
    """
    cleaned = path.strip("/").replace("{", "").replace("}", "")
    return cleaned.replace("/", "_") or "root"


def _join_descriptions(*parts: str | None) -> str:
    """Join non-empty description parts with " — "."""
    return " — ".join(p for p in parts if p)


def _operation_to_tool(method: str, path: str, op: dict[str, Any]) -> Tool:
    """Convert one OpenAPI operation into a Tool."""
    tool_id = op.get("operationId") or f"{method.lower()}_{_path_to_id(path)}"
    description = _join_descriptions(op.get("summary"), op.get("description"))

    properties: dict[str, Any] = {}
    required: list[str] = []

    # Parameters (path / query / header / cookie).
    for param in op.get("parameters", []) or []:
        pname = param.get("name")
        if not pname:
            continue
        pschema = dict(param.get("schema") or {})
        if "description" not in pschema and param.get("description"):
            pschema["description"] = param["description"]
        properties[pname] = pschema
        if param.get("required"):
            required.append(pname)

    # Request body. We only look at application/json content.
    body = op.get("requestBody") or {}
    body_content = body.get("content") or {}
    json_content = body_content.get("application/json") or {}
    body_schema = json_content.get("schema")
    if body_schema:
        if body_schema.get("type") == "object":
            # Merge body properties into the tool's flat parameter list.
            # Same pattern most LLM tool callers use - one flat namespace.
            for k, v in (body_schema.get("properties") or {}).items():
                properties[k] = v
            required.extend(body_schema.get("required") or [])
        else:
            # Non-object body (array, primitive). Nest under "body".
            properties["body"] = body_schema
            if body.get("required"):
                required.append("body")

    parameters_schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        # Dedupe while preserving order.
        seen: set[str] = set()
        dedup_required: list[str] = []
        for r in required:
            if r not in seen:
                seen.add(r)
                dedup_required.append(r)
        parameters_schema["required"] = dedup_required

    keywords = list(op.get("tags") or [])

    return Tool(
        id=tool_id,
        name=tool_id,
        description=description,
        parameters_schema=parameters_schema,
        keywords=keywords,
    )


def _resolve_refs(spec: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$ref`` references via jsonref.

    jsonref returns a lazy-proxy object; we deep-copy it into a plain dict
    so downstream code (json.dumps, deepcopy, etc.) doesn't trip on the
    proxy.
    """
    try:
        import jsonref
    except ImportError as exc:
        raise ImportError(
            "OpenAPISource requires the 'openapi' extra. Install with:\n"
            "    pip install 'toolpicker[openapi]'"
        ) from exc
    import json

    resolved = jsonref.replace_refs(spec, proxies=False)
    # jsonref's output may still contain shared references; round-trip
    # through JSON to get a clean plain dict tree.
    return json.loads(json.dumps(resolved))  # type: ignore[no-any-return]


def _validate(spec: dict[str, Any]) -> None:
    """Validate against the OpenAPI spec. Raises if invalid."""
    try:
        from openapi_spec_validator import validate
    except ImportError as exc:
        raise ImportError(
            "OpenAPISource requires the 'openapi' extra. Install with:\n"
            "    pip install 'toolpicker[openapi]'"
        ) from exc
    validate(spec)


class OpenAPISource:
    """Wrap an OpenAPI 3.0 / 3.1 spec as a ``ToolSource``.

    Args:
        spec: The spec, either as a parsed dict or a path to a YAML/JSON
            file. YAML files require ``pyyaml``; JSON files have no extra
            deps.
        validate: Whether to run ``openapi-spec-validator``. Default
            ``True``. Set to ``False`` to skip validation for trusted specs
            or to handle minor non-conformances gracefully.
    """

    def __init__(
        self,
        spec: dict[str, Any] | str | Path,
        *,
        validate: bool = True,
    ) -> None:
        spec_dict = self._load(spec)
        if validate:
            _validate(spec_dict)
        resolved = _resolve_refs(spec_dict)
        self._tools = self._extract_tools(resolved)
        ids = [t.id for t in self._tools]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(
                f"duplicate tool ids from OpenAPI source: {sorted(dupes)}; "
                "set explicit operationId values to disambiguate"
            )

    @staticmethod
    def _load(spec: dict[str, Any] | str | Path) -> dict[str, Any]:
        if isinstance(spec, dict):
            return spec
        path = Path(spec)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "Loading YAML specs requires PyYAML. Install with:\n    pip install pyyaml"
                ) from exc
            loaded: Any = yaml.safe_load(text)
        else:
            import json

            loaded = json.loads(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"spec at {path} did not parse to a dict")
        return loaded

    @staticmethod
    def _extract_tools(resolved: dict[str, Any]) -> list[Tool]:
        tools: list[Tool] = []
        paths = resolved.get("paths") or {}
        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue
            for method in _HTTP_METHODS:
                op = path_item.get(method)
                if isinstance(op, dict):
                    tools.append(_operation_to_tool(method, path, op))
        return tools

    def tools(self) -> list[Tool]:
        return list(self._tools)
