# Protocol Mapping

The report in `Guide.md` describes conceptual JSON-RPC messages. The implementation maps those concepts onto the official MCP surface.

## MCP transport

- Transport: Streamable HTTP.
- Server shape: `FastMCP(..., stateless_http=True, json_response=True)`.
- Client shape: `ClientSession` over `streamable_http_client(...)`.

## Implemented tools

- `match_skills`
  - Input: anomaly context, entropy, tags, and current candidate labels.
  - Output: best-matching skill manifests and prompt patches.

- `reflect_anomaly`
  - Input: anomaly case, baseline perception, and optional already-applied skills.
  - Output: corrected scene graph, reflection summary, and a compiled skill manifest.

## Implemented resource

- `skill://{skill_id}`
  - Returns the skill manifest and the associated markdown note in one payload.

## Why not raw custom JSON-RPC

The conceptual messages in the report are valid design notes, but using MCP tools/resources directly keeps the implementation compliant with current MCP server and client SDKs.
