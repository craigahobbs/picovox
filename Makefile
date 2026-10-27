# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE


# The published site (make gh-pages) - the API documentation (build/doc/), the paper (paper/, at /paper/), and
# make train's pinned book texts (the bundle data/nn/books.json records, at /books/)
GHPAGES_SRC := build/site/


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


# The site: the documentation, the paper, and the books bundle beside it
.PHONY: site
site: doc
	rm -rf build/site
	mkdir -p build/site
	rsync -a build/doc/ build/site/
	rsync -a paper/ build/site/paper/
	PYTHONPATH=train/src python3 -B -m picovox_train.books publish build/site

_gh-pages: site


# The development tools are linted too; the generated model modules (lib/models/) aren't
ESLINT_ARGS := $(ESLINT_ARGS) tools/ nn/js/ --ignore-pattern 'lib/models/**'


# The voice lab (tools/voicelab/) staged with the package's lib/ in build/voicelab/ - published as an artifact (Claude's
# questions and the answers in its db store; index.html the page, the rest its files) or served locally (Explore only).
# VOICELAB_TRAINED - a directory of trained models (make train-check's TRAIN_DIR), staged beside the shipped ones as
# build/voicelab/trained/ (its g2pModel.js and voiceModel.js)
VOICELAB_TRAINED ?=

.PHONY: voicelab
voicelab:
	rm -rf build/voicelab
	mkdir -p build/voicelab/picovox
	rsync -a tools/voicelab/ build/voicelab/
	rsync -a lib build/voicelab/picovox/
ifneq '$(VOICELAB_TRAINED)' ''
	mkdir -p build/voicelab/trained
	cp $(VOICELAB_TRAINED)/g2pModel.js $(VOICELAB_TRAINED)/voiceModel.js build/voicelab/trained/
endif


# The training package (train/ - a python-build package, picovox_train): make commit runs its gate too (its unit tests,
# pylint, and 100% branch coverage - make -C train commit), and make clean cleans it
.PHONY: commit-train
commit: commit-train
commit-train: parity
	$(MAKE) -C train commit


# The package's outputs of the parity checks' inputs (tools/parity.js - build/parity-outputs.json), which the training
# package's tests check its Python port against
.PHONY: parity
parity:
	$(NODE_SHELL) node tools/parity.js


# Python virtual environments for make train - build/venv-<name> from requirements/<name>.txt (for example,
# build/venv-listeners.build for the listener-model panel, train/src/picovox_train/listeners.py); they run the training
# package from its source (PYTHONPATH=train/src - TRAIN_SRC). PYTHON_VENV is the Python that creates them (3.12 through
# 3.14 are known to work). CMAKE_BUILD_PARALLEL_LEVEL (default, every core) is the source builds' jobs - Praat
# (parselmouth) on Linux ARM, which has no wheel, otherwise builds serially.
PYTHON_VENV ?= python3
CMAKE_BUILD_PARALLEL_LEVEL ?= $(shell getconf _NPROCESSORS_ONLN)

build/venv-%.build: requirements/%.txt
	rm -rf build/venv-$*
	$(PYTHON_VENV) -m venv build/venv-$*
	build/venv-$*/bin/pip -q install --upgrade pip
	CMAKE_BUILD_PARALLEL_LEVEL=$(CMAKE_BUILD_PARALLEL_LEVEL) build/venv-$*/bin/pip -q install -r requirements/$*.txt
	touch $@

.PHONY: listeners
listeners: build/venv-listeners.build


# make train - both generated models from a clean checkout, by the recipe data/nn/train.json, trained to their gates in
# TRAIN_DIR (TRAIN_DIR/g2pModel.js and voiceModel.js - lib/models/ unchanged): make train-g2p (the G2P and the heteronym
# relabeling, CPU) and make train-voice (the voice - its data, teacher, and student, GPU) at once, then the report (each
# model's size and scores, and whether it differs from the committed module - picovox_train.trained report). The gates:
# the G2P - identical to the committed module below its header, else within the recipe's tolerances; the voice - no
# worse than the shipped model on the listener panel. Each stage is a stamp in TRAIN_DIR; make -j train runs the
# independent stages at once - the GPU stages one at a time (picovox_train/gpulock.py: two at once each run at about
# half speed), the CPU stages beside them (the G2P, the data builds, the evaluations, and the listening samples). The
# voice data is built with the committed G2P (the G2P is deterministic, so its training reproduces it; a G2P installed
# that differs makes the next make train rebuild the voice).
#
# The flow: make train; the training dashboard (its gates and listening samples); make train-check (the package's tests
# with the trained models, and the voice lab staged with them for the blind A/B against the shipped models); then make
# train-install (the trained models that differ into lib/models/). make train-check and train-install fail until make
# train has trained both models.
TRAIN_DIR ?= build/train
TRAIN_SRC := $(CURDIR)/train/src
NN_PYTHON := env PYTHONPATH=$(TRAIN_SRC) build/venv-nn/bin/python -B
LISTENERS_PYTHON := env PYTHONPATH=$(TRAIN_SRC) build/venv-listeners/bin/python -B
POS_PYTHON := env PYTHONPATH=$(TRAIN_SRC) build/venv-pos/bin/python -B
TRAIN_ENV := NN_DATA_DIR=$(TRAIN_DIR)/data NN_RUNS_DIR=$(TRAIN_DIR)/runs NN_DISTILL_DIR=$(TRAIN_DIR)/distill

# The CPU stages that run beside a GPU stage (the G2P chain, the heteronym check, the teacher's evaluation, the
# samples) at the utility QoS: CPU work beside a GPU stage shares the unified memory's bandwidth - three memory-bound
# jobs at the default QoS slowed teacher B 53%, at utility 15%, at background 1% (but 4.6x slower themselves)
BESIDE := taskpolicy -c utility
# (the GPU stages hold the GPU lock - the machine's, whatever the TRAIN_DIR - and use two CPU threads: more don't speed a
# step, and the CPU stages beside them would oversubscribe the cores)
GPU := OMP_NUM_THREADS=2 $(NN_PYTHON) -m picovox_train.gpulock build/gpu.lock --
TRAIN_RUNS := $(TRAIN_DIR)/runs

# The bookkeeping, for the training dashboard (tools/dashboard/): each stage's start, end, and failure in
# $(TRAIN_DIR)/stages.jsonl and each command's output also in $(TRAIN_DIR)/logs/<stage>.log (tools/trainstage.sh - the
# teacher's output also in its run's train.log), and the CPU, GPU, and memory use with the running stages every 20 seconds
# in $(TRAIN_DIR)/util.log (tools/trainutil.sh, for the life of make train). The stages, their dependencies, and their
# expected times are data/nn/train-stages.json's (test/testTrainStages.js checks it against these targets).
STAGE = tools/trainstage.sh $(TRAIN_DIR) $(basename $(@F))

# Record a make target's run in stages.jsonl (the training dashboard shows its stages), and keep the Mac awake while make
# runs: macOS's caffeinate until make exits - finished, failed, or interrupted - on AC power with the lid closed too (-s;
# idle sleep paused one run's teacher B for 28 minutes); elsewhere, nothing
TRAIN_TARGET = @mkdir -p $(TRAIN_DIR) && printf '{"event": "target", "target": "%s", "time": %s}\n' $(1) "$$(date +%s)" >> $(TRAIN_DIR)/stages.jsonl; \
	if command -v caffeinate >/dev/null; then nohup caffeinate -i -s -w $$PPID >/dev/null 2>&1 & fi

# The targets' final stages - the voice's (its package, gate, and listening samples) and the G2P's (its gate and the
# heteronym relabeling's)
TRAIN_VOICE := $(TRAIN_DIR)/package.done $(TRAIN_DIR)/student-eval.done $(TRAIN_DIR)/samples-reference.done \
	$(TRAIN_DIR)/samples-teacher-a.done $(TRAIN_DIR)/samples-teacher-b.done $(TRAIN_DIR)/samples-student.done \
	$(TRAIN_DIR)/samples-polish.done
TRAIN_G2P := $(TRAIN_DIR)/g2p.done $(TRAIN_DIR)/heteronyms.done
TRAIN_REPORT := $(NN_PYTHON) -m picovox_train.trained report $(TRAIN_DIR)

# Fail unless make train has trained both models (make train-check and train-install)
TRAINED = @test -f $(TRAIN_DIR)/package.done -a -f $(TRAIN_DIR)/g2p.done || \
	{ echo 'make $@: no trained models in $(TRAIN_DIR) - make train first'; exit 1; }

# (the power check first, alone - its prompt has the terminal - then the stages, in parallel with make -j, then the report)
.PHONY: train train-voice train-g2p train-check train-install train-util train-clean
train:
	$(call TRAIN_TARGET,train)
	@tools/trainpower.sh
	@$(MAKE) --no-print-directory train-util $(TRAIN_VOICE) $(TRAIN_G2P)
	@$(TRAIN_REPORT)

train-voice:
	$(call TRAIN_TARGET,train-voice)
	@tools/trainpower.sh
	@$(MAKE) --no-print-directory train-util $(TRAIN_VOICE)
	@$(TRAIN_REPORT)

# make train-g2p - the G2P alone, on every core (G2P_STANDALONE_JOBS workers at the default QoS, not after the readers'
# data build)
train-g2p:
	$(call TRAIN_TARGET,train-g2p)
	@$(MAKE) --no-print-directory train-util G2P_STANDALONE=1 G2P_JOBS=$(G2P_STANDALONE_JOBS) CONTEXT_JOBS=$(G2P_STANDALONE_JOBS) BESIDE= $(TRAIN_G2P)
	@$(TRAIN_REPORT)

# make train-check - the trained models checked before they're installed: the package's tests with them in place of
# lib/models/'s (tools/trainmodels.js - lib/models/ unchanged), and the voice lab staged with them (build/voicelab/trained/)
# for the blind A/B against the shipped models
train-check:
	$(TRAINED)
	NODE_OPTIONS='--import=$(CURDIR)/tools/trainmodels.js' TRAIN_MODELS=$(TRAIN_DIR) node --test $(NODE_TEST_ARGS)
	@$(MAKE) --no-print-directory voicelab VOICELAB_TRAINED=$(TRAIN_DIR)
	@echo 'make train-check: the package'"'"'s tests pass with the trained models; the voice lab is staged in build/voicelab/ with them' \
		'(trained) beside the shipped models - ask Claude for the blind A/B, then make train-install to install them'

# make train-install - the trained models that differ from the committed ones into lib/models/
train-install:
	$(TRAINED)
	$(NN_PYTHON) -m picovox_train.trained install $(TRAIN_DIR)

# The utilization sampler - it exits when make does (finished, failed, or interrupted)
train-util:
	@mkdir -p $(TRAIN_DIR) && nohup tools/trainutil.sh $(TRAIN_DIR) $$PPID >/dev/null 2>&1 &

# A from-scratch make train: remove TRAIN_DIR (make train itself never removes anything)
train-clean:
	@case '$(TRAIN_DIR)' in ''|/|.|..) echo 'make train-clean: TRAIN_DIR is "$(TRAIN_DIR)" - not removed'; exit 1;; esac
	@echo 'make train-clean: removing $(TRAIN_DIR) - every make train stage stamp, its data, runs, evaluations, and logs'
	rm -rf $(TRAIN_DIR)

# The corpora - CMU ARCTIC, LibriTTS-R and its alignments, CREMA-D (downloads, once)
$(TRAIN_DIR)/corpora.done: | build/venv-nn.build
	@$(STAGE) start
	$(STAGE) -- build/venv-nn/bin/python -I train/src/picovox_train/corpora.py
	$(STAGE) -- $(NN_PYTHON) -m picovox_train.cremad fetch
	@$(STAGE) end
	@mkdir -p $(@D) && touch $@

# The G2P (CPU): the network's four stages (one core, deterministic), the heteronym context labels (the books - pinned
# texts, data/nn/books.json - and part-of-speech tags in build/venv-pos), the context network and its held-out accuracy
# gate (cross-validated over the aligned speakers) beside the G2P network, then the reading stage (two threads), the
# export to $(TRAIN_DIR)/g2pModel.js, and picovox_train.context check against the committed module; and the heteronym
# relabeling, which must reproduce data/nn/g2p-relabel.json (the G2P network's input).
#
# In make train, the G2P's memory-bound stages (the context labels, the relabeling) are scheduled after the readers'
# data build (order-only prerequisites) - into the teacher's GPU-bound hours, when the cores are idle - not competing
# with the data build on the critical path; the G2P network itself (one core) starts at once. G2P_JOBS is the context
# labels' worker processes - the aligned and tag workers; the outputs are the same for any count. make train runs the
# chain beside the GPU stages, so its count is low (BESIDE) for the memory-bound context labels; the context network's
# trainings (its four teachers, and each of cross-validate's ten folds' four teachers and network - one thread each,
# small and compute-bound) take CONTEXT_JOBS (at 2, the folds outlasted the voice stages by 40 minutes); make train-g2p
# runs G2P_STANDALONE_JOBS for both.
G2P_JOBS ?= 2
CONTEXT_JOBS ?= 6
G2P_STANDALONE_JOBS ?= 10

$(TRAIN_DIR)/g2p-network.done: data/nn/train.json data/nn/g2p-words.json data/nn/g2p-relabel.json data/nn/g2p-fixes.json data/nn/g2p-books.json \
		data/cmudict.dict \
		| build/venv-nn.build
	@$(STAGE) start
	$(STAGE) -- $(NN_PYTHON) -m picovox_train.g2p train --out $(TRAIN_DIR)/g2p
	@$(STAGE) end
	@mkdir -p $(@D) && touch $@

$(TRAIN_DIR)/context-labels.done: $(TRAIN_DIR)/corpora.done data/nn/train.json data/nn/books.json data/nn/context-labels.json \
		data/nn/g2p-fixes.json data/nn/g2p-books.json \
		data/nn/heteronyms-test.json | build/venv-nn.build build/venv-pos.build $(if $(G2P_STANDALONE),,$(TRAIN_DIR)/data-multi.done)
	@$(STAGE) start
	$(STAGE) -- $(BESIDE) $(NN_PYTHON) -m picovox_train.contextlabels books
	$(STAGE) -- $(BESIDE) $(NN_PYTHON) -m picovox_train.contextlabels aligned --out $(TRAIN_DIR)/context -j $(G2P_JOBS)
	$(STAGE) -- $(BESIDE) $(POS_PYTHON) -m picovox_train.contextlabels tag --out $(TRAIN_DIR)/context -j $(G2P_JOBS)
	$(STAGE) -- $(BESIDE) $(NN_PYTHON) -m picovox_train.contextlabels labels --out $(TRAIN_DIR)/context
	@$(STAGE) end
	touch $@

# (its log new - the training dashboard counts the log's finished trainings as the stage's progress)
$(TRAIN_DIR)/context-network.done: $(TRAIN_DIR)/context-labels.done
	@$(STAGE) start
	$(STAGE) -- rm -f $(TRAIN_DIR)/context/train.log
	$(STAGE) -- $(BESIDE) $(NN_PYTHON) -m picovox_train.context train-context --labels $(TRAIN_DIR)/context/labels.json --out $(TRAIN_DIR)/context/context.pt \
		--log $(TRAIN_DIR)/context/train.log -j $(CONTEXT_JOBS)
	$(STAGE) -- $(BESIDE) $(NN_PYTHON) -m picovox_train.context cross-validate --labels $(TRAIN_DIR)/context/labels.json --out $(TRAIN_DIR)/context/folds \
		--log $(TRAIN_DIR)/context/train.log -j $(CONTEXT_JOBS)
	@$(STAGE) end
	touch $@

$(TRAIN_DIR)/g2p.done: $(TRAIN_DIR)/g2p-network.done $(TRAIN_DIR)/context-labels.done $(TRAIN_DIR)/context-network.done
	@$(STAGE) start
	$(STAGE) -- $(BESIDE) $(NN_PYTHON) -m picovox_train.context train-g2p --labels $(TRAIN_DIR)/context/labels.json --init $(TRAIN_DIR)/g2p/focus.pt \
		--out $(TRAIN_DIR)/context/g2p.pt --log $(TRAIN_DIR)/context/train.log
	$(STAGE) -- $(BESIDE) $(NN_PYTHON) -m picovox_train.context export --g2p $(TRAIN_DIR)/context/g2p.pt --context $(TRAIN_DIR)/context/context.pt \
		--out $(TRAIN_DIR)/g2pModel.js
	$(STAGE) -- $(BESIDE) $(NN_PYTHON) -m picovox_train.context check $(TRAIN_DIR)/g2pModel.js
	@$(STAGE) end
	touch $@


# The heteronym relabeling (CPU) - it must reproduce data/nn/g2p-relabel.json's words
$(TRAIN_DIR)/heteronyms.done: $(TRAIN_DIR)/corpora.done data/cmudict.dict data/nn/g2p-relabel.json \
		| $(if $(G2P_STANDALONE),,$(TRAIN_DIR)/data-multi.done)
	@$(STAGE) start
	$(STAGE) -- $(BESIDE) $(NN_PYTHON) -m picovox_train.heteronyms --out $(TRAIN_DIR)/heteronyms -j 4 --check data/nn/g2p-relabel.json
	@$(STAGE) end
	touch $@

# The training data - the readers (CPU, every core; the pool and its analysis settings are data/nn/speakers.json's) and
# the emotional speech (its alignment on the GPU - at the default QoS: at the utility QoS it took twice as long, and
# teacher A, which needs its speakers, waited 3 minutes for it)
$(TRAIN_DIR)/data-multi.done: $(TRAIN_DIR)/corpora.done lib/models/g2pModel.js data/nn/speakers.json
	@$(STAGE) start
	$(STAGE) -- $(TRAIN_ENV) $(NN_PYTHON) -m picovox_train.multi build --out $(TRAIN_DIR)/data/multi --pool data/nn/speakers.json
	@$(STAGE) end
	touch $@

$(TRAIN_DIR)/cremad-align.done: $(TRAIN_DIR)/corpora.done | build/venv-listeners.build
	@$(STAGE) start
	$(STAGE) -- $(GPU) $(LISTENERS_PYTHON) -m picovox_train.cremad align
	@$(STAGE) end
	touch $@

$(TRAIN_DIR)/data-cremad.done: $(TRAIN_DIR)/cremad-align.done lib/models/g2pModel.js
	@$(STAGE) start
	$(STAGE) -- $(TRAIN_ENV) $(NN_PYTHON) -m picovox_train.cremad build --out $(TRAIN_DIR)/data/cremad
	@$(STAGE) end
	touch $@

# The teacher - stage A, then stage B with the critic (GPU; resumable), and its evaluation (CPU)
$(TRAIN_DIR)/teacher-a.done: $(TRAIN_DIR)/data-multi.done $(TRAIN_DIR)/data-cremad.done data/nn/train.json
	@$(STAGE) start
	$(STAGE) --log $(TRAIN_RUNS)/teacher-a/train.log -- \
		$(TRAIN_ENV) $(GPU) $(NN_PYTHON) -m picovox_train.trainmulti --name teacher-a --recipe teacher.a --resume
	@$(STAGE) end
	touch $@

$(TRAIN_DIR)/teacher-b.done: $(TRAIN_DIR)/teacher-a.done
	@$(STAGE) start
	$(STAGE) --log $(TRAIN_RUNS)/teacher-b/train.log -- \
		$(TRAIN_ENV) $(GPU) $(NN_PYTHON) -m picovox_train.trainmulti --name teacher-b --recipe teacher.b --init $(TRAIN_RUNS)/teacher-a/last.pt --resume
	@$(STAGE) end
	touch $@

$(TRAIN_DIR)/teacher-eval.done: $(TRAIN_DIR)/teacher-b.done
	@$(STAGE) start
	$(STAGE) -- $(TRAIN_ENV) $(BESIDE) $(NN_PYTHON) -m picovox_train.distill eval teacher=$(TRAIN_RUNS)/teacher-b/best.pt --whisper-workers 3
	@$(STAGE) end
	touch $@

# The student - the distillation targets, the student, its polish (GPU), and the gate (CPU: no worse than the shipped
# lib/models/voiceModel.js - evaluated as packaged, the baseline - within the recipe's tolerance, and its voice similar
# to the teacher's; a better teacher never fails it). The baseline is scored as soon as the teacher's evaluation is done
# (its similarity is to the teacher's voice), beside the student's GPU stages - not on the critical path after the polish.
$(TRAIN_DIR)/targets.done: $(TRAIN_DIR)/teacher-b.done
	@$(STAGE) start
	$(STAGE) -- $(TRAIN_ENV) $(GPU) $(NN_PYTHON) -m picovox_train.distill targets --teacher $(TRAIN_RUNS)/teacher-b/best.pt
	@$(STAGE) end
	touch $@

$(TRAIN_DIR)/student.done: $(TRAIN_DIR)/targets.done
	@$(STAGE) start
	$(STAGE) -- $(TRAIN_ENV) $(GPU) $(NN_PYTHON) -m picovox_train.distill train --name student
	@$(STAGE) end
	touch $@

$(TRAIN_DIR)/polish.done: $(TRAIN_DIR)/student.done
	@$(STAGE) start
	$(STAGE) -- $(TRAIN_ENV) $(GPU) $(NN_PYTHON) -m picovox_train.distill polish --init $(TRAIN_RUNS)/distill-student/best.pt --name student-polish \
		--teacher $(TRAIN_RUNS)/teacher-b/best.pt
	@$(STAGE) end
	touch $@

$(TRAIN_DIR)/baseline-eval.done: $(TRAIN_DIR)/teacher-eval.done
	@$(STAGE) start
	$(STAGE) -- $(TRAIN_ENV) $(BESIDE) $(NN_PYTHON) -m picovox_train.distill eval baseline=lib/models/voiceModel.js --whisper-workers 3
	@$(STAGE) end
	touch $@

$(TRAIN_DIR)/student-eval.done: $(TRAIN_DIR)/polish.done $(TRAIN_DIR)/baseline-eval.done
	@$(STAGE) start
	$(STAGE) -- $(TRAIN_ENV) $(NN_PYTHON) -m picovox_train.distill eval student=$(TRAIN_RUNS)/distill-student-polish/best.pt --whisper-workers 3
	$(STAGE) -- $(TRAIN_ENV) $(NN_PYTHON) -m picovox_train.distill pick --no-samples --against baseline
	@$(STAGE) end
	touch $@

# The voice model module, TRAIN_DIR/voiceModel.js, with its provenance (the commit, the recipe's and the data's hashes,
# the gate's scores)
$(TRAIN_DIR)/package.done: $(TRAIN_DIR)/student-eval.done
	@$(STAGE) start
	$(STAGE) -- $(TRAIN_ENV) $(NN_PYTHON) -m picovox_train.export $(TRAIN_RUNS)/distill-student-polish/best.pt --voice multi --out $(TRAIN_DIR)/export
	$(STAGE) -- $(TRAIN_ENV) $(NN_PYTHON) -m picovox_train.package $(TRAIN_DIR)/export/multi -o $(TRAIN_DIR)/voiceModel.js --provenance --recipe data/nn/train.json \
		--data $(TRAIN_DIR)/data/multi $(TRAIN_DIR)/data/cremad --scores $(TRAIN_DIR)/distill/eval/student/result.json
	@$(STAGE) end
	touch $@


# The listening samples (CPU, each beside the next GPU stage - tools/trainsamples.js): each training stage's final model
# (the checkpoint the next stage starts from, exported as the package stage exports it) speaking data/nn/train.json's
# samples texts in its speakers' voices as the package speaks them, scored with UTMOS, in $(TRAIN_DIR)/samples/ (index.json)
# for the training dashboard's Listen tab - and the reference, the shipped lib/models/voiceModel.js (in the teacher's
# GPU-bound hours, after the readers' data build; again when the shipped model changes)
SAMPLES := OMP_NUM_THREADS=2 $(BESIDE) node tools/trainsamples.js --out $(TRAIN_DIR)/samples

$(TRAIN_DIR)/samples-reference.done: lib/models/voiceModel.js | build/venv-listeners.build $(TRAIN_DIR)/data-multi.done
	@$(STAGE) start
	$(STAGE) -- $(SAMPLES) --stage reference --model lib/models/voiceModel.js
	@$(STAGE) end
	touch $@

$(TRAIN_DIR)/samples-teacher-a.done: $(TRAIN_DIR)/teacher-a.done | build/venv-listeners.build
	@$(STAGE) start
	$(STAGE) -- $(TRAIN_ENV) $(SAMPLES) --stage teacher-a --checkpoint $(TRAIN_RUNS)/teacher-a/last.pt
	@$(STAGE) end
	touch $@

$(TRAIN_DIR)/samples-teacher-b.done: $(TRAIN_DIR)/teacher-b.done | build/venv-listeners.build
	@$(STAGE) start
	$(STAGE) -- $(TRAIN_ENV) $(SAMPLES) --stage teacher-b --checkpoint $(TRAIN_RUNS)/teacher-b/best.pt
	@$(STAGE) end
	touch $@

$(TRAIN_DIR)/samples-student.done: $(TRAIN_DIR)/student.done | build/venv-listeners.build
	@$(STAGE) start
	$(STAGE) -- $(TRAIN_ENV) $(SAMPLES) --stage student --checkpoint $(TRAIN_RUNS)/distill-student/best.pt
	@$(STAGE) end
	touch $@

$(TRAIN_DIR)/samples-polish.done: $(TRAIN_DIR)/polish.done | build/venv-listeners.build
	@$(STAGE) start
	$(STAGE) -- $(TRAIN_ENV) $(SAMPLES) --stage polish --checkpoint $(TRAIN_RUNS)/distill-student-polish/best.pt
	@$(STAGE) end
	touch $@


clean:
	rm -rf Makefile.base jsdoc.json eslint.config.js
	$(MAKE) -C train clean
