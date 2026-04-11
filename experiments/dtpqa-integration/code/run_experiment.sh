#!/usr/bin/env bash
#
# Large-scale automated experiment runner for DTPQA real dataset
# Following autoresearch two-loop architecture principles
#

set -euo pipefail

cd "$(dirname "$0")/../.."  # Go to project root

# Default configuration
MODE="${MODE:-baseline}"
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-1}"
MCP_PORT="${MCP_PORT:-8003}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Python environment
    if ! command -v uv &> /dev/null; then
        log_error "uv not found. Please install uv."
        exit 1
    fi

    # Check data directory
    if [ ! -d "data/dtpqa" ]; then
        log_error "DTPQA data directory not found at data/dtpqa"
        exit 1
    fi

    # Check MCP server
    if ! curl -s "http://127.0.0.1:${MCP_PORT}/mcp" > /dev/null 2>&1; then
        log_warn "MCP server not running on port ${MCP_PORT}"
        log_info "Please start the MCP server first:"
        log_info "  uv run ad-mcp-server"
        read -p "Start MCP server now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Starting MCP server..."
            uv run ad-mcp-server &
            MCP_PID=$!
            sleep 3
            if ! curl -s "http://127.0.0.1:${MCP_PORT}/mcp" > /dev/null 2>&1; then
                log_error "Failed to start MCP server"
                exit 1
            fi
            log_success "MCP server started (PID: $MCP_PID)"
        else
            exit 1
        fi
    else
        log_success "MCP server is running on port ${MCP_PORT}"
    fi

    # Check skill store
    SKILL_STORE="/tmp/dtpqa_skills_empty"
    if [ -d "$SKILL_STORE" ] && [ "$(ls -A $SKILL_STORE)" ]; then
        log_warn "Skill store is not empty: $SKILL_STORE"
        read -p "Clean skill store? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$SKILL_STORE"/*
            log_success "Skill store cleaned"
        fi
    fi
}

# Show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Large-scale automated experiment runner for DTPQA real dataset

OPTIONS:
    --mode MODE         Experiment mode: baseline, reflection, ablation, comprehensive
                        (default: $MODE)
    --limit N           Limit to N samples (for testing)
    --no-resume         Don't resume from checkpoint
    --watch RUN_ID      Monitor a running experiment
    --report RUN_IDS    Generate report from completed runs (space-separated)
    --help              Show this help

EXAMPLES:
    # Run baseline experiment
    $0 --mode baseline

    # Run reflection experiment with 100 samples
    $0 --mode reflection --limit 100

    # Run comprehensive batch
    $0 --mode comprehensive

    # Monitor experiment
    $0 --watch run-dtpqa-real-baseline-20260331

    # Generate report
    $0 --report run1 run2 run3

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --no-resume)
            RESUME="0"
            shift
            ;;
        --watch)
            WATCH_RUN_ID="$2"
            shift 2
            ;;
        --report)
            shift
            REPORT_RUN_IDS=("$@")
            break
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    if [ -n "${WATCH_RUN_ID:-}" ]; then
        log_info "Monitoring experiment: $WATCH_RUN_ID"
        uv run python experiments/dtpqa-integration/code/run_large_scale_experiments.py --watch "$WATCH_RUN_ID"
        exit 0
    fi

    if [ -n "${REPORT_RUN_IDS:-}" ]; then
        log_info "Generating report for: ${REPORT_RUN_IDS[*]}"
        uv run python experiments/dtpqa-integration/code/run_large_scale_experiments.py \
            --mode report \
            --run-ids "${REPORT_RUN_IDS[@]}"
        exit 0
    fi

    log_info "================================"
    log_info "AUTORESEARCH Experiment Runner"
    log_info "================================"
    log_info "Mode: $MODE"
    log_info "Dataset: DTPQA real (ONLY)"
    [ -n "$LIMIT" ] && log_info "Limit: $LIMIT samples"
    log_info "Resume: $([ "$RESUME" = "1" ] && echo 'yes' || echo 'no')"
    log_info ""

    check_prerequisites

    # Build command
    CMD="uv run python experiments/dtpqa-integration/code/run_large_scale_experiments.py --mode $MODE"
    [ -n "$LIMIT" ] && CMD="$CMD --limit $LIMIT"
    [ "$RESUME" = "0" ] && CMD="$CMD --no-resume"

    log_info "Starting experiment..."
    log_info "Command: $CMD"
    echo

    eval $CMD

    log_success "Experiment complete!"
}

main "$@"
