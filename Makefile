ifneq ($(filter setup-dev,$(MAKECMDGOALS)),)
ISOEXEC_ROOT ?= $(HOME)/.isoexec-dev
else
ISOEXEC_ROOT ?= $(HOME)/.isoexec
endif
WHEEL ?=
SETUP_FLAGS ?=
ARGS ?=
# Any Python 3.12 runs setup; the venv's exact Python comes from environments/default.toml.
SETUP := uv run --no-project --python 3.12 python scripts/setup_env.py --root "$(ISOEXEC_ROOT)" $(SETUP_FLAGS)

DIST_DIR ?= dist
BUILD_DIR ?= build
RELEASE_TAG ?=

# Empty: check-wheel uses uv's Python 3.12 (the supported version) and check-public-api uses python3.
CHECK_PYTHON ?=

.DEFAULT_GOAL := help
.PHONY: help build check-wheel check-public-api setup setup-dev check-env run-tiny
help: ## Show the targets
	@if [ -t 1 ] && [ -z "$$NO_COLOR" ]; then c='\033[36m'; b='\033[1m'; d='\033[2m'; r='\033[0m'; fi; \
	awk -v c="$$c" -v b="$$b" -v d="$$d" -v r="$$r" 'BEGIN {FS = ":.*## "; \
		printf "%sUsage:%s make %s<target>%s %s[ISOEXEC_ROOT=dir] [WHEEL=path]%s\n\n%sTargets:%s\n", b, r, c, r, d, r, b, r} \
		/^[a-zA-Z_-]+:.*## / {printf "  %s%-18s%s %s\n", c, $$1, r, $$2}' $(MAKEFILE_LIST)

build: ## Build committed HEAD into new directories; no GPU/Torch required
	python3 scripts/build_package.py --out "$(DIST_DIR)" --source "$(BUILD_DIR)/src" $(if $(RELEASE_TAG),--tag "$(RELEASE_TAG)",)

check-wheel: ## Check package files, installation and import isoexec (no Torch needed)
	$(if $(CHECK_PYTHON),"$(CHECK_PYTHON)",uv run --no-project --python 3.12 python) scripts/check_wheel.py \
		$(DIST_DIR)/isoexec-*.whl --source "$(BUILD_DIR)/src"

check-public-api: ## Test the exported API and meta model construction (requires Torch)
	"$(or $(CHECK_PYTHON),python3)" scripts/check_wheel.py $(DIST_DIR)/isoexec-*.whl --public-api

setup: ## Download the release wheel and create or reuse the managed runtime
	$(SETUP) $(if $(WHEEL),--wheel "$(WHEEL)",)

setup-dev: ## Install editable IsoExec and dev tools (default ~/.isoexec-dev); also checks [[metadata_conflicts]]
	$(SETUP) --dev

check-env: ## Check the existing managed runtime without installing packages
	$(SETUP) --check-only

run-tiny: setup ## Train Qwen3.5-0.8B for one step on one GPU; ARGS passes SkyRL key=value overrides
	uv run --no-project --python 3.12 python examples/run_tiny.py --root "$(ISOEXEC_ROOT)" $(ARGS)
