# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE


# Download python-build
PYTHON_BUILD_DIR ?= ../../python-build
define WGET
ifeq '$$(wildcard $(notdir $(1)))' ''
$$(info Downloading $(notdir $(1)))
$$(shell [ -f $(PYTHON_BUILD_DIR)/$(notdir $(1)) ] && cp $(PYTHON_BUILD_DIR)/$(notdir $(1)) . || $(call WGET_CMD, $(1)))
endif
endef
WGET_CMD = if command -v wget >/dev/null 2>&1; then wget -q -c $(1); else curl -f -Os $(1); fi
$(eval $(call WGET, https://craigahobbs.github.io/python-build/Makefile.base))
$(eval $(call WGET, https://craigahobbs.github.io/python-build/pylintrc))


# The Pythons the training code supports - its pinned numpy and torch install on 3.12 through 3.14 (set before the
# include, which filters them)
PYTHON_IMAGES_EXCLUDE := python:3.11 python:3.15


# Include python-build
include Makefile.base


# PARITY_WRITE=1 rewrites the parity checks' inputs (src/tests/parity.py) - in the containers too
DOCKER_ENV := $(if $(PARITY_WRITE),-e PARITY_WRITE=1)


# The code's style documents each function with a short comment above it (CLAUDE.md), not a docstring
PYLINT_ARGS := $(PYLINT_ARGS) --disable=missing-function-docstring


clean:
	rm -rf Makefile.base pylintrc
