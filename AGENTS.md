# AGENTS.md

This repository is a sibling of `/Users/thomastupper/projects/foxlight/Skulk`.

## Purpose

Use this repo for Skulk test harness tooling, scenario runners, integration
fixtures, and validation utilities that should live outside the main Skulk
source tree.

## Working Notes

- Treat Skulk itself as the source of truth for runtime behavior, commands, and
  architecture.
- Prefer small, reproducible harnesses over ad hoc scripts.
- Keep generated logs, caches, and local run artifacts out of git.

