#!/usr/bin/env python3
"""Container-portable entry point for the audited IKET provider tree."""

from iket.cli.main import entrypoint


if __name__ == "__main__":
    raise SystemExit(entrypoint())
