# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE


# The published site - the app (web/), its library (lib/), the data, and the catalog
GHPAGES_SRC := build/site/

# jsdom - a DOM for testing the book parsers (DOMParser) and for building the catalog
USE_JSDOM := 1


# Download javascript-build
JAVASCRIPT_BUILD_DIR ?= ../javascript-build
define WGET
ifeq '$$(wildcard $(notdir $(1)))' ''
$$(info Downloading $(notdir $(1)))
$$(shell [ -f $(JAVASCRIPT_BUILD_DIR)/$(notdir $(1)) ] && cp $(JAVASCRIPT_BUILD_DIR)/$(notdir $(1)) . || $(call WGET_CMD, $(1)))
endif
endef
WGET_CMD = if command -v wget >/dev/null 2>&1; then wget -q -c $(1); else curl -f -Os $(1); fi
$(eval $(call WGET, https://craigahobbs.github.io/javascript-build/Makefile.base))
$(eval $(call WGET, https://craigahobbs.github.io/javascript-build/jsdoc.json))
$(eval $(call WGET, https://craigahobbs.github.io/javascript-build/eslint.config.js))


# Include javascript-build
include Makefile.base


ESLINT_ARGS := $(ESLINT_ARGS) web/ tools/


# The book catalog - every Standard Ebooks book and the top GUTENBERG_COUNT Project Gutenberg books (remote pages are
# cached in out/catalog/, so a rebuild resumes; set GITHUB_TOKEN for GitHub's higher rate limits)
GUTENBERG_COUNT ?= 1000
CATALOG_ARGS ?=

.PHONY: catalog
catalog: build/npm.build
	$(NODE_SHELL) node tools/catalog.js --gutenberg $(GUTENBERG_COUNT)$(if $(CATALOG_ARGS), $(CATALOG_ARGS))

data/catalog.json:
	$(MAKE) catalog


# Python virtual environments for the analysis tools - build/venv-<name> from requirements/<name>.txt (for example,
# build/venv-listeners.build for the listener-model panel, tools/listeners.py). PYTHON_VENV is the Python that creates
# them (3.12 through 3.14 are known to work).
PYTHON_VENV ?= python3

build/venv-%.build: requirements/%.txt
	rm -rf build/venv-$*
	$(PYTHON_VENV) -m venv build/venv-$*
	build/venv-$*/bin/pip -q install --upgrade pip
	build/venv-$*/bin/pip -q install -r requirements/$*.txt
	touch $@

.PHONY: listeners
listeners: build/venv-listeners.build


# The published site - the app, its library, the paper, the neural voices, the data, and the catalog
.PHONY: site
site: data/catalog.json
	rm -rf build/site
	mkdir -p build/site/data
	rsync -a --exclude .DS_Store index.html README.md web lib paper build/site/
	cp data/LICENSE.cmudict data/catalog.json build/site/data/
	mkdir -p build/site/nn build/site/data/nn
	rsync -a --exclude .DS_Store nn/js nn/vocab.json build/site/nn/
	cp data/nn/multi.js build/site/data/nn/

_gh-pages: site


clean:
	rm -rf Makefile.base jsdoc.json eslint.config.js
