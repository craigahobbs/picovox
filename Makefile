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
# builds, and the evaluations). The voice data is built with the committed G2P; the G2P stages retrain it beside the
# voice stages and check the result against it (they never replace it).
TRAIN_DIR ?= build/train
NN_PYTHON := build/venv-nn/bin/python -B
LISTENERS_PYTHON := build/venv-listeners/bin/python -B
TRAIN_ENV := NN_DATA_DIR=$(TRAIN_DIR)/data NN_RUNS_DIR=$(TRAIN_DIR)/runs NN_DISTILL_DIR=$(TRAIN_DIR)/distill
# (the GPU stages hold the GPU lock and use two CPU threads - more don't speed a step, and the CPU stages beside them
# would oversubscribe the cores)
GPU := OMP_NUM_THREADS=2 $(NN_PYTHON) -m nn.gpulock $(TRAIN_DIR)/gpu.lock --
TRAIN_RUNS := $(TRAIN_DIR)/runs

.PHONY: train
train: $(TRAIN_DIR)/g2p.done $(TRAIN_DIR)/heteronyms.done $(TRAIN_DIR)/package.done

# The corpora - CMU ARCTIC, LibriTTS-R and its alignments, CREMA-D (downloads, once)
$(TRAIN_DIR)/corpora.done: | build/venv-analysis.build build/venv-nn.build
	build/venv-analysis/bin/python -I tools/corpora.py
	$(NN_PYTHON) -m nn.cremad fetch
	@mkdir -p $(@D) && touch $@

# The G2P (CPU) - a check beside the voice stages, which use the committed lib/models/g2pModel.js: the network's four
# stages (one core, deterministic), the heteronym context labels (the books fetched by id; part-of-speech tags in
# build/venv-pos), the reading stage and the context network (two threads each), the export to $(TRAIN_DIR), the
# held-out accuracy gate (the context network cross-validated over the aligned speakers), and nn.context check: identical
# to the committed module, else within the recipe's tolerances (the committed module is kept either way - make train
# never replaces it)
$(TRAIN_DIR)/g2p-network.done: data/nn/train.json data/nn/g2p-words.json data/nn/g2p-relabel.json data/cmudict.dict | build/venv-nn.build
	$(NN_PYTHON) -m nn.g2p train --out $(TRAIN_DIR)/g2p
	@mkdir -p $(@D) && touch $@

$(TRAIN_DIR)/context-labels.done: $(TRAIN_DIR)/corpora.done data/nn/train.json data/nn/context-labels.json \
		data/nn/heteronyms-test.json | build/venv-nn.build build/venv-pos.build
	$(NN_PYTHON) -m nn.contextlabels books
	$(NN_PYTHON) -m nn.contextlabels aligned --out $(TRAIN_DIR)/context -j 3
	build/venv-pos/bin/python -B -m nn.contextlabels tag --out $(TRAIN_DIR)/context -j 3
	$(NN_PYTHON) -m nn.contextlabels labels --out $(TRAIN_DIR)/context
	touch $@

$(TRAIN_DIR)/g2p.done: $(TRAIN_DIR)/g2p-network.done $(TRAIN_DIR)/context-labels.done
	$(NN_PYTHON) -m nn.context train-g2p --labels $(TRAIN_DIR)/context/labels.json --init $(TRAIN_DIR)/g2p/focus.pt \
		--out $(TRAIN_DIR)/context/g2p.pt --log $(TRAIN_DIR)/context/train.log
	$(NN_PYTHON) -m nn.context train-context --labels $(TRAIN_DIR)/context/labels.json --out $(TRAIN_DIR)/context/context.pt \
		--log $(TRAIN_DIR)/context/train.log
	$(NN_PYTHON) -m nn.context export --g2p $(TRAIN_DIR)/context/g2p.pt --context $(TRAIN_DIR)/context/context.pt \
		--out $(TRAIN_DIR)/g2pModel.js
	$(NN_PYTHON) -m nn.context cross-validate --labels $(TRAIN_DIR)/context/labels.json --out $(TRAIN_DIR)/context/folds \
		--log $(TRAIN_DIR)/context/train.log
	$(NN_PYTHON) -m nn.context check $(TRAIN_DIR)/g2pModel.js
	touch $@


# The heteronym relabeling (CPU) - it must reproduce data/nn/g2p-relabel.json's words
$(TRAIN_DIR)/heteronyms.done: $(TRAIN_DIR)/corpora.done
	$(NN_PYTHON) -m nn.heteronyms --out $(TRAIN_DIR)/heteronyms -j 4 --check data/nn/g2p-relabel.json
	touch $@

# The training data - the readers (CPU, every core; the pool and its analysis settings are data/nn/speakers.json's) and
# the emotional speech (its alignment on the GPU)
$(TRAIN_DIR)/data-multi.done: $(TRAIN_DIR)/corpora.done lib/models/g2pModel.js data/nn/speakers.json
	$(TRAIN_ENV) $(NN_PYTHON) -m nn.multi build --out $(TRAIN_DIR)/data/multi --pool data/nn/speakers.json
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
