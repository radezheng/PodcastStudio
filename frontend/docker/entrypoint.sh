#!/usr/bin/env sh
set -eu

API_BASE="${PODCASTSTUDIO_API_BASE:-}"

# Escape backslashes and quotes for safe JS string literal.
ESCAPED_API_BASE=$(printf '%s' "$API_BASE" | sed 's/\\/\\\\/g; s/"/\\"/g')

cat > /usr/share/nginx/html/config.js <<EOF
// Generated at container startup.
window.__PODCASTSTUDIO_CONFIG__ = {
  apiBase: "${ESCAPED_API_BASE}",
};
EOF

exec nginx -g 'daemon off;'
