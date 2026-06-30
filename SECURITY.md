# Security Policy

## Reporting a Vulnerability

Please report security issues privately to the maintainers rather than opening
a public issue with exploit details. Include the affected version or commit,
the impact, and reproduction steps when possible.

## Operational Safety

Some stability commands can intentionally kill or relaunch Skulk processes on
remote nodes. These commands require an explicit `--execute-destructive` flag,
and contributors should keep destructive defaults out of public configs.
