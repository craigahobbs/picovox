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


# The training package (train/ - a python-build package, picovox_train): make commit runs its gate too (its unit tests,
# pylint, and 100% branch coverage - make -C train commit), and make clean cleans it
.PHONY: commit-train
commit: commit-train
commit-train:
	$(MAKE) -C train commit


# Python virtual environments for make train - build/venv-<name> from requirements/<name>.txt (for example,
# build/venv-listeners.build for the listener-model panel, train/src/picovox_train/listeners.py); they run the training
# package from its source (PYTHONPATH=train/src - TRAIN_SRC). PYTHON_VENV is the Python that creates them (3.12 through
# 3.14 are known to work).
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
# stages one at a time (picovox_train/gpulock.py: two at once each run at about half speed), the CPU stages beside them (the
# G2P retrain and the heteronym relabeling - each checked against the committed file it reproduces - the data
# builds, and the evaluations). The voice data is built with the committed G2P; the G2P stages retrain it beside the
# voice stages and check the result against it (they never replace it).
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
# (the GPU stages hold the GPU lock and use two CPU threads - more don't speed a step, and the CPU stages beside them
# would oversubscribe the cores)
GPU := OMP_NUM_THREADS=2 $(NN_PYTHON) -m picovox_train.gpulock $(TRAIN_DIR)/gpu.lock --
TRAIN_RUNS := $(TRAIN_DIR)/runs

# The bookkeeping, for the training dashboard (tools/dashboard/): each stage's start, end, and failure in
# $(TRAIN_DIR)/stages.jsonl and each command's output also in $(TRAIN_DIR)/logs/<stage>.log (tools/trainstage.sh - the
# teacher's output also in its run's train.log), and the CPU, GPU, and memory use with the running stages every 20 seconds
# in $(TRAIN_DIR)/util.log (tools/trainutil.sh, for the life of make train). The stages, their dependencies, and their
# expected times are data/nn/train-stages.json's (test/testTrainStages.js checks it against these targets).
STAGE = tools/trainstage.sh $(TRAIN_DIR) $(basename $(@F))

# Record a make target's run in stages.jsonl (the training dashboard shows its stages)
TRAIN_TARGET = @mkdir -p $(TRAIN_DIR) && printf '{"event": "target", "target": "%s", "time": %s}\n' $(1) "$$(date +%s)" >> $(TRAIN_DIR)/stages.jsonl

.PHONY: train train-stages train-util train-clean
train:
	$(call TRAIN_TARGET,train)
	@tools/trainpower.sh
	@$(MAKE) --no-print-directory train-stages

# (the power check first, alone - its prompt has the terminal - then the stages, in parallel with make -j)
train-stages: train-util $(TRAIN_DIR)/package.done $(TRAIN_DIR)/samples-reference.done \
	$(TRAIN_DIR)/samples-teacher-a.done $(TRAIN_DIR)/samples-teacher-b.done $(TRAIN_DIR)/samples-student.done $(TRAIN_DIR)/samples-polish.done

# The utilization sampler - it exits when make does (finished, failed, or interrupted)
train-util:
	@mkdir -p $(TRAIN_DIR) && nohup tools/trainutil.sh $(TRAIN_DIR) $$PPID >/dev/null 2>&1 &

# A from-scratch make train: remove TRAIN_DIR (make train itself never removes anything)
train-clean:
	@case '$(TRAIN_DIR)' in ''|/|.|..) echo 'make train-clean: TRAIN_DIR is "$(TRAIN_DIR)" - not removed'; exit 1;; esac
	@echo 'make train-clean: removing $(TRAIN_DIR) - every make train stage stamp, its data, runs, evaluations, and logs'
	rm -rf $(TRAIN_DIR)

# The corpora - CMU ARCTIC, LibriTTS-R and its alignments, CREMA-D (downloads, once)
$(TRAIN_DIR)/corpora.done: | build/venv-analysis.build build/venv-nn.build
	@$(STAGE) start
	$(STAGE) -- build/venv-analysis/bin/python -I train/src/picovox_train/corpora.py
	$(STAGE) -- $(NN_PYTHON) -m picovox_train.cremad fetch
	@$(STAGE) end
	@mkdir -p $(@D) && touch $@

# The CPU-only checks (the G2P chain, the heteronym relabeling) are scheduled after the readers' data build (order-only
# prerequisites) - into the teacher's GPU-bound hours, when the cores are idle - not competing with the data build on
# the critical path; the G2P network itself needs no corpora, so it runs during their download.
#
# The G2P (CPU) - a check beside the voice stages, which use the committed lib/models/g2pModel.js: the network's four
# stages (one core, deterministic), the heteronym context labels (the books - pinned texts, data/nn/books.json - and
# part-of-speech tags in build/venv-pos), the context network and its held-out accuracy gate (cross-validated over the
# aligned speakers) beside the G2P network, then the reading stage (two threads), the export to $(TRAIN_DIR), and
# picovox_train.context check: identical to the committed module, else within the recipe's tolerances (the committed
# module is kept either way - make train never replaces it).
#
# G2P_JOBS is the chain's worker processes - the context labels' aligned and tag workers and cross-validate's folds at
# once (each fold one thread); the outputs are the same for any count. make train runs the chain beside the GPU stages,
# so its count is low (BESIDE). make -j train-g2p runs the G2P chain alone, for standalone G2P work: G2P_STANDALONE_JOBS
# workers at the default QoS, and not after the readers' data build.
G2P_JOBS ?= 2
G2P_STANDALONE_JOBS ?= 10

# make train-check - the checks that make train's committed inputs reproduce: the G2P chain (the committed
# lib/models/g2pModel.js, identical below its header, else within the recipe's tolerances) and the heteronym relabeling
# (data/nn/g2p-relabel.json's words) - standalone, on every core, so they never compete with make train's GPU stages for
# memory bandwidth (beside teacher A they cost it about 9 minutes). Run it when the G2P's inputs change.
.PHONY: train-check
train-check:
	$(call TRAIN_TARGET,train-check)
	@$(MAKE) --no-print-directory G2P_STANDALONE=1 G2P_JOBS=$(G2P_STANDALONE_JOBS) BESIDE= $(TRAIN_DIR)/g2p.done $(TRAIN_DIR)/heteronyms.done

.PHONY: train-g2p
train-g2p:
	$(call TRAIN_TARGET,train-g2p)
	@$(MAKE) --no-print-directory G2P_STANDALONE=1 G2P_JOBS=$(G2P_STANDALONE_JOBS) BESIDE= $(TRAIN_DIR)/g2p.done

$(TRAIN_DIR)/g2p-network.done: data/nn/train.json data/nn/g2p-words.json data/nn/g2p-relabel.json data/nn/g2p-fixes.json data/cmudict.dict \
		| build/venv-nn.build
	@$(STAGE) start
	$(STAGE) -- $(NN_PYTHON) -m picovox_train.g2p train --out $(TRAIN_DIR)/g2p
	@$(STAGE) end
	@mkdir -p $(@D) && touch $@

$(TRAIN_DIR)/context-labels.done: $(TRAIN_DIR)/corpora.done data/nn/train.json data/nn/books.json data/nn/context-labels.json \
		data/nn/g2p-fixes.json \
		data/nn/heteronyms-test.json | build/venv-nn.build build/venv-pos.build $(if $(G2P_STANDALONE),,$(TRAIN_DIR)/data-multi.done)
	@$(STAGE) start
	$(STAGE) -- $(BESIDE) $(NN_PYTHON) -m picovox_train.contextlabels books
	$(STAGE) -- $(BESIDE) $(NN_PYTHON) -m picovox_train.contextlabels aligned --out $(TRAIN_DIR)/context -j $(G2P_JOBS)
	$(STAGE) -- $(BESIDE) $(POS_PYTHON) -m picovox_train.contextlabels tag --out $(TRAIN_DIR)/context -j $(G2P_JOBS)
	$(STAGE) -- $(BESIDE) $(NN_PYTHON) -m picovox_train.contextlabels labels --out $(TRAIN_DIR)/context
	@$(STAGE) end
	touch $@

$(TRAIN_DIR)/context-network.done: $(TRAIN_DIR)/context-labels.done
	@$(STAGE) start
	$(STAGE) -- $(BESIDE) $(NN_PYTHON) -m picovox_train.context train-context --labels $(TRAIN_DIR)/context/labels.json --out $(TRAIN_DIR)/context/context.pt \
		--log $(TRAIN_DIR)/context/train.log -j $(G2P_JOBS)
	$(STAGE) -- $(BESIDE) $(NN_PYTHON) -m picovox_train.context cross-validate --labels $(TRAIN_DIR)/context/labels.json --out $(TRAIN_DIR)/context/folds \
		--log $(TRAIN_DIR)/context/train.log -j $(G2P_JOBS)
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
$(TRAIN_DIR)/heteronyms.done: $(TRAIN_DIR)/corpora.done
	@$(STAGE) start
	$(STAGE) -- $(BESIDE) $(NN_PYTHON) -m picovox_train.heteronyms --out $(TRAIN_DIR)/heteronyms -j 4 --check data/nn/g2p-relabel.json
	@$(STAGE) end
	touch $@

# The training data - the readers (CPU, every core; the pool and its analysis settings are data/nn/speakers.json's) and
# the emotional speech (its alignment on the GPU)
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
# to the teacher's; a better teacher never fails it)
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

$(TRAIN_DIR)/student-eval.done: $(TRAIN_DIR)/polish.done $(TRAIN_DIR)/teacher-eval.done
	@$(STAGE) start
	$(STAGE) -- $(TRAIN_ENV) $(NN_PYTHON) -m picovox_train.distill eval student=$(TRAIN_RUNS)/distill-student-polish/best.pt \
		baseline=lib/models/voiceModel.js --whisper-workers 3
	$(STAGE) -- $(TRAIN_ENV) $(NN_PYTHON) -m picovox_train.distill pick --no-samples --against baseline
	@$(STAGE) end
	touch $@

# The voice model module, with its provenance (the commit, the recipe's and the data's hashes, the gate's scores) - after
# the reference samples, which hear the module it replaces
$(TRAIN_DIR)/package.done: $(TRAIN_DIR)/student-eval.done | $(TRAIN_DIR)/samples-reference.done
	@$(STAGE) start
	$(STAGE) -- $(TRAIN_ENV) $(NN_PYTHON) -m picovox_train.export $(TRAIN_RUNS)/distill-student-polish/best.pt --voice multi --out $(TRAIN_DIR)/export
	$(STAGE) -- $(TRAIN_ENV) $(NN_PYTHON) -m picovox_train.package $(TRAIN_DIR)/export/multi -o lib/models/voiceModel.js --provenance --recipe data/nn/train.json \
		--data $(TRAIN_DIR)/data/multi $(TRAIN_DIR)/data/cremad --scores $(TRAIN_DIR)/distill/eval/student/result.json
	@$(STAGE) end
	touch $@


# The listening samples (CPU, each beside the next GPU stage - tools/trainsamples.js): each training stage's final model
# (the checkpoint the next stage starts from, exported as the package stage exports it) speaking data/nn/train.json's
# samples texts in its speakers' voices as the package speaks them, scored with UTMOS, in $(TRAIN_DIR)/samples/ (index.json)
# for the training dashboard's Listen tab - and the reference, the shipped lib/models/voiceModel.js before the package
# stage replaces it (in the teacher's GPU-bound hours, after the readers' data build)
SAMPLES := OMP_NUM_THREADS=2 $(BESIDE) node tools/trainsamples.js --out $(TRAIN_DIR)/samples

$(TRAIN_DIR)/samples-reference.done: | build/venv-listeners.build $(TRAIN_DIR)/data-multi.done
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
