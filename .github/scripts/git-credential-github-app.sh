#!/bin/bash
# Git credential helper that generates a fresh GitHub App installation token.
#
# Generates a JWT from the app private key, exchanges it for an installation
# access token via the GitHub API. This avoids token expiry issues during
# long-running benchmark jobs.
#
# Required environment variables:
#   APP_ID        - GitHub App ID
#   APP_KEY_FILE  - Path to the app private key PEM file
#
# Usage:
#   git config credential.helper \
#     '!APP_ID=123 APP_KEY_FILE=/tmp/key.pem /path/to/git-credential-github-app.sh'

set -euo pipefail

# Only handle "get" operations
if [ "${1:-}" != "get" ]; then
    exit 0
fi

# Read stdin to verify this is for github.com
HOST=""
while IFS='=' read -r key value; do
    case "$key" in
        host) HOST="$value" ;;
    esac
done

if [ "$HOST" != "github.com" ]; then
    exit 0
fi

# --- Generate JWT ---

NOW=$(date +%s)
IAT=$((NOW - 60))
EXP=$((NOW + 600))

b64url() { openssl base64 -A | tr '+/' '-_' | tr -d '='; }

HEADER=$(printf '{"alg":"RS256","typ":"JWT"}' | b64url)
PAYLOAD=$(printf '{"iat":%d,"exp":%d,"iss":"%s"}' "$IAT" "$EXP" "$APP_ID" | b64url)
SIGNATURE=$(printf '%s.%s' "$HEADER" "$PAYLOAD" \
    | openssl dgst -sha256 -sign "$APP_KEY_FILE" -binary | b64url)

JWT="${HEADER}.${PAYLOAD}.${SIGNATURE}"

# --- Exchange JWT for installation token ---

REPO="${GITHUB_REPOSITORY:-}"
if [ -z "$REPO" ]; then
    REPO=$(git remote get-url origin \
        | sed -n 's|.*github\.com[:/]\([^.]*\)\(\.git\)\{0,1\}/*$|\1|p')
fi

INSTALL_ID=$(curl -sf \
    -H "Authorization: Bearer $JWT" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/${REPO}/installation" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

TOKEN=$(curl -sf -X POST \
    -H "Authorization: Bearer $JWT" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/app/installations/${INSTALL_ID}/access_tokens" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")

# Output in git credential format
echo "protocol=https"
echo "host=github.com"
echo "username=x-access-token"
echo "password=${TOKEN}"
