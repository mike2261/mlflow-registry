#!/usr/bin/env bash
# Manage the per-model serving containers next to the registry stack.
#
#   scripts/serve.sh build-base              build mlflow-serving-base (CUDA + torch + mlflow), once
#   scripts/serve.sh build NAME...           build the image(s) for NAME
#   scripts/serve.sh up NAME...              start NAME (builds if missing); host port from the catalog
#   scripts/serve.sh down NAME...            stop and remove NAME's container
#   scripts/serve.sh logs NAME               follow NAME's logs
#   scripts/serve.sh ps                      running serving containers with ports
#   scripts/serve.sh health NAME             GET /health on NAME's port
#   scripts/serve.sh names                   list servable model names
#
# NAME is a registered model name, e.g. kokoro-82m. Run from any directory.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
COMPOSE=(docker compose -f docker-compose.yaml -f docker-compose.serving.yaml)

names() { grep -oE '^  serve-[a-z0-9.-]+:' docker-compose.serving.yaml | sed -E 's/^  serve-//; s/:$//'; }

port_of() {
  # host port published for NAME, read from the generated compose file
  awk -v svc="  serve-$1:" '
    $0 == svc { found = 1 }
    found && /^      - "[0-9]+:8080"/ { gsub(/[^0-9:]/, "", $2); split($2, p, ":"); print p[1]; exit }
  ' docker-compose.serving.yaml
}

require_names() { [ "$#" -gt 0 ] || { echo "usage: $0 $cmd NAME..." >&2; exit 2; }; }

cmd=${1:-help}
shift || true

case "$cmd" in
  build-base) "${COMPOSE[@]}" --profile build build serving-base ;;
  build)  require_names "$@"; for n in "$@"; do "${COMPOSE[@]}" --profile "$n" build "serve-$n"; done ;;
  up)     require_names "$@"; for n in "$@"; do "${COMPOSE[@]}" --profile "$n" up -d "serve-$n"; echo "serve-$n -> http://localhost:$(port_of "$n")/invocations"; done ;;
  down)   require_names "$@"; for n in "$@"; do "${COMPOSE[@]}" --profile "$n" stop "serve-$n"; "${COMPOSE[@]}" --profile "$n" rm -f "serve-$n"; done ;;
  logs)   require_names "$@"; "${COMPOSE[@]}" --profile "$1" logs -f "serve-$1" ;;
  ps)     docker ps --filter "name=mr-serve-" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' ;;
  health) require_names "$@"; curl -fsS "http://localhost:$(port_of "$1")/health" && echo ;;
  names)  names ;;
  help|-h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) echo "unknown command: $cmd" >&2; exit 2 ;;
esac
