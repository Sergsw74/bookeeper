#!/usr/bin/env bash
# ==============================================================================
# scripts/build_docker.sh
# Build and publish multi-architecture Docker image with automatic version & release date stamping
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

REGISTRY="192.168.50.20:5000/bookeeper"
PLATFORMS="linux/amd64,linux/arm64"
PUSH=true
BUMP=false
CUSTOM_VERSION=""
CUSTOM_DATE="$(date +%Y-%m-%d)"

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --version <X.Y.Z>   Explicit version to build (default: from pyproject.toml)"
    echo "  --bump              Bump patch version (e.g., 0.1.0 -> 0.1.1)"
    echo "  --date <YYYY-MM-DD> Explicit release date (default: today: ${CUSTOM_DATE})"
    echo "  --registry <URL>    Docker registry repository (default: ${REGISTRY})"
    echo "  --platform <list>   Target platforms (default: ${PLATFORMS})"
    echo "  --no-push           Do not push image to registry"
    echo "  -h, --help          Show this help message"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)
            CUSTOM_VERSION="$2"
            shift 2
            ;;
        --bump)
            BUMP=true
            shift
            ;;
        --date)
            CUSTOM_DATE="$2"
            shift 2
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --platform)
            PLATFORMS="$2"
            shift 2
            ;;
        --no-push)
            PUSH=false
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

STAMP_ARGS=()
if [ "$BUMP" = true ]; then
    STAMP_ARGS+=(--bump)
elif [ -n "$CUSTOM_VERSION" ]; then
    STAMP_ARGS+=(--version "$CUSTOM_VERSION")
fi
STAMP_ARGS+=(--date "$CUSTOM_DATE")

# Update repository files
python3 "${SCRIPT_DIR}/stamp_version.py" "${STAMP_ARGS[@]}"

# Get the stamped version
VERSION=$(python3 "${SCRIPT_DIR}/stamp_version.py" --get)
RELEASE_DATE="$CUSTOM_DATE"

echo "Building Docker Image: ${REGISTRY}:${VERSION} (release_date: ${RELEASE_DATE})"

# Build Docker command
BUILD_CMD=(docker buildx build)
BUILD_CMD+=(--platform "${PLATFORMS}")
BUILD_CMD+=(--build-arg "APP_VERSION=${VERSION}")
BUILD_CMD+=(--build-arg "RELEASE_DATE=${RELEASE_DATE}")
BUILD_CMD+=(-t "${REGISTRY}:latest")
BUILD_CMD+=(-t "${REGISTRY}:${VERSION}")

if [ "$PUSH" = true ]; then
    BUILD_CMD+=(--push)
else
    BUILD_CMD+=(--load)
fi

BUILD_CMD+=(.)

echo "Executing: ${BUILD_CMD[*]}"
"${BUILD_CMD[@]}"

echo "✓ Successfully built and published ${REGISTRY}:${VERSION} (${RELEASE_DATE})"
