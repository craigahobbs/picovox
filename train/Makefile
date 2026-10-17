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


# Include python-build
include Makefile.base


# The code's style documents each function with a short comment above it (CLAUDE.md), not a docstring
PYLINT_ARGS := $(PYLINT_ARGS) --disable=missing-function-docstring


clean:
	rm -rf Makefile.base pylintrc
