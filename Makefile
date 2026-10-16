# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE


# The published site - the app (static/audiobook/), its library (lib/), the data, and the catalog
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


# The generated model modules (lib/models/) aren't linted
ESLINT_ARGS := $(ESLINT_ARGS) static/audiobook/ tools/ --ignore-pattern 'lib/models/**'


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


# make train - every generated model (lib/models/g2pModel.js and voiceModel.js) from a clean checkout, by the recipe
# data/nn/train.json. Each stage is a stamp in TRAIN_DIR; make -j train runs the independent stages at once - the GPU
# stages one at a time (nn/gpulock.py: two at once each run at about half speed), the CPU stages beside them (the
# G2P retrain and the heteronym relabeling - each checked against the committed file it reproduces - the data
# builds, and the evaluations). The voice data is built with the committed G2P; if the retrained G2P differs, its
# stage updates lib/models/g2pModel.js and fails, and make train again rebuilds the data with it.
TRAIN_DIR ?= build/train
NN_PYTHON := build/venv-nn/bin/python -B
LISTENERS_PYTHON := build/venv-listeners/bin/python -B
TRAIN_ENV := NN_DATA_DIR=$(TRAIN_DIR)/data NN_RUNS_DIR=$(TRAIN_DIR)/runs NN_DISTILL_DIR=$(TRAIN_DIR)/distill
GPU := $(NN_PYTHON) -m nn.gpulock $(TRAIN_DIR)/gpu.lock --
TRAIN_RUNS := $(TRAIN_DIR)/runs

.PHONY: train
train: $(TRAIN_DIR)/g2p.done $(TRAIN_DIR)/heteronyms.done $(TRAIN_DIR)/package.done

# The corpora - CMU ARCTIC, LibriTTS-R and its alignments, CREMA-D (downloads, once)
$(TRAIN_DIR)/corpora.done: | build/venv-analysis.build build/venv-nn.build
	build/venv-analysis/bin/python -I tools/corpora.py
	$(NN_PYTHON) -m nn.cremad fetch
	@mkdir -p $(@D) && touch $@

# The G2P retrain (one CPU core, deterministic) - it must reproduce lib/models/g2pModel.js
$(TRAIN_DIR)/g2p.done: data/nn/train.json data/nn/g2p-words.json data/nn/g2p-relabel.json data/cmudict.dict | build/venv-nn.build
	$(NN_PYTHON) -m nn.g2p train --out $(TRAIN_DIR)/g2p
	$(NN_PYTHON) -m nn.g2p export $(TRAIN_DIR)/g2p --out $(TRAIN_DIR)/g2pModel.js
	@cmp -s $(TRAIN_DIR)/g2pModel.js lib/models/g2pModel.js || { cp $(TRAIN_DIR)/g2pModel.js lib/models/g2pModel.js; \
		echo 'make train: the retrained G2P differs - lib/models/g2pModel.js updated; make train again rebuilds the voice data'; exit 1; }
	touch $@

# The heteronym relabeling (CPU) - it must reproduce data/nn/g2p-relabel.json's words
$(TRAIN_DIR)/heteronyms.done: $(TRAIN_DIR)/corpora.done
	$(NN_PYTHON) -m nn.heteronyms --out $(TRAIN_DIR)/heteronyms -j 4 --check data/nn/g2p-relabel.json
	touch $@

# The reader pool - accents and per-speaker analysis settings (CPU)
$(TRAIN_DIR)/pool.done: $(TRAIN_DIR)/corpora.done | build/venv-listeners.build
	$(LISTENERS_PYTHON) -m nn.accent
	$(NN_PYTHON) -m nn.multi settings
	touch $@

# The training data - the readers (CPU, every core) and the emotional speech (its alignment on the GPU)
$(TRAIN_DIR)/data-multi.done: $(TRAIN_DIR)/pool.done lib/models/g2pModel.js
	$(TRAIN_ENV) $(NN_PYTHON) -m nn.multi build --out $(TRAIN_DIR)/data/multi
	touch $@

$(TRAIN_DIR)/cremad-align.done: $(TRAIN_DIR)/corpora.done | build/venv-listeners.build
	$(GPU) $(LISTENERS_PYTHON) -m nn.cremad align
	touch $@

$(TRAIN_DIR)/data-cremad.done: $(TRAIN_DIR)/cremad-align.done lib/models/g2pModel.js
	$(TRAIN_ENV) $(NN_PYTHON) -m nn.cremad build --out $(TRAIN_DIR)/data/cremad
	touch $@

# The teacher - stage A, then stage B with the critic (GPU; resumable), and its evaluation (CPU)
$(TRAIN_DIR)/teacher-a.done: $(TRAIN_DIR)/data-multi.done $(TRAIN_DIR)/data-cremad.done data/nn/train.json
	$(TRAIN_ENV) $(GPU) $(NN_PYTHON) -m nn.trainmulti --name teacher-a --recipe teacher.a --resume
	touch $@

$(TRAIN_DIR)/teacher-b.done: $(TRAIN_DIR)/teacher-a.done
	$(TRAIN_ENV) $(GPU) $(NN_PYTHON) -m nn.trainmulti --name teacher-b --recipe teacher.b --init $(TRAIN_RUNS)/teacher-a/last.pt --resume
	touch $@

$(TRAIN_DIR)/teacher-eval.done: $(TRAIN_DIR)/teacher-b.done
	$(TRAIN_ENV) $(NN_PYTHON) -m nn.distill eval teacher=$(TRAIN_RUNS)/teacher-b/best.pt --whisper-workers 3
	touch $@

# The student - the distillation targets, the student, its polish (GPU), and the gate (CPU: within the tolerance of
# the teacher)
$(TRAIN_DIR)/targets.done: $(TRAIN_DIR)/teacher-b.done
	$(TRAIN_ENV) $(GPU) $(NN_PYTHON) -m nn.distill targets --teacher $(TRAIN_RUNS)/teacher-b/best.pt
	touch $@

$(TRAIN_DIR)/student.done: $(TRAIN_DIR)/targets.done
	$(TRAIN_ENV) $(GPU) $(NN_PYTHON) -m nn.distill train --name student
	touch $@

$(TRAIN_DIR)/polish.done: $(TRAIN_DIR)/student.done
	$(TRAIN_ENV) $(GPU) $(NN_PYTHON) -m nn.distill polish --init $(TRAIN_RUNS)/distill-student/best.pt --name student-polish \
		--teacher $(TRAIN_RUNS)/teacher-b/best.pt
	touch $@

$(TRAIN_DIR)/student-eval.done: $(TRAIN_DIR)/polish.done $(TRAIN_DIR)/teacher-eval.done
	$(TRAIN_ENV) $(NN_PYTHON) -m nn.distill eval student=$(TRAIN_RUNS)/distill-student-polish/best.pt --whisper-workers 3
	$(TRAIN_ENV) $(NN_PYTHON) -m nn.distill pick --no-samples
	touch $@

# The voice model module, with its provenance (the commit, the recipe's and the data's hashes, the gate's scores)
$(TRAIN_DIR)/package.done: $(TRAIN_DIR)/student-eval.done
	$(NN_PYTHON) -m nn.export $(TRAIN_RUNS)/distill-student-polish/best.pt --voice multi --out $(TRAIN_DIR)/export
	$(NN_PYTHON) -m nn.package $(TRAIN_DIR)/export/multi -o lib/models/voiceModel.js --provenance --recipe data/nn/train.json \
		--data $(TRAIN_DIR)/data/multi $(TRAIN_DIR)/data/cremad --scores $(TRAIN_DIR)/distill/eval/student/result.json
	touch $@


# The published site - the app, its library, the paper, the neural voices, the data, and the catalog
.PHONY: site
site: data/catalog.json
	rm -rf build/site
	mkdir -p build/site/data
	rsync -a --exclude .DS_Store index.html README.md static lib paper build/site/
	cp data/LICENSE.cmudict data/catalog.json build/site/data/
	mkdir -p build/site/nn
	rsync -a --exclude .DS_Store nn/js nn/vocab.json build/site/nn/

_gh-pages: site


clean:
	rm -rf Makefile.base jsdoc.json eslint.config.js
