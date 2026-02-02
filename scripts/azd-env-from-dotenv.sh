#!/usr/bin/env bash
set -euo pipefail

DOTENV_PATH="${1:-.env}"

if [[ ! -f "$DOTENV_PATH" ]]; then
  echo "dotenv file not found: $DOTENV_PATH" >&2
  exit 1
fi

while IFS= read -r line || [[ -n "$line" ]]; do
  # Trim leading/trailing whitespace
  line="${line#${line%%[![:space:]]*}}"
  line="${line%${line##*[![:space:]]}}"

  # Skip blanks and comments
  [[ -z "$line" ]] && continue
  [[ "$line" == \#* ]] && continue

  # Skip non-assignment lines
  [[ "$line" != *"="* ]] && continue

  key="${line%%=*}"
  value="${line#*=}"

  # Trim again
  key="${key#${key%%[![:space:]]*}}"
  key="${key%${key##*[![:space:]]}}"
  value="${value#${value%%[![:space:]]*}}"
  value="${value%${value##*[![:space:]]}}"

  # Strip surrounding single/double quotes
  if [[ "$value" =~ ^\".*\"$ ]]; then
    value="${value:1:${#value}-2}"
  elif [[ "$value" =~ ^\'.*\'$ ]]; then
    value="${value:1:${#value}-2}"
  fi

  if [[ -n "$key" ]]; then
    azd env set "$key" "$value"
  fi

done < "$DOTENV_PATH"
