"""CLI for the MCP server."""

from __future__ import annotations

import typer

from ad_cornercase.bootstrap import build_cloud_service, load_settings
from ad_cornercase.logging import configure_logging
from ad_cornercase.mcp.server import create_mcp_server

app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def run() -> None:
    runtime_settings, project_settings = load_settings()
    configure_logging(runtime_settings.log_level)
    service = build_cloud_service(runtime_settings, project_settings)
    server = create_mcp_server(
        service=service,
        runtime_settings=runtime_settings,
        project_settings=project_settings,
    )
    server.run(transport="streamable-http")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
