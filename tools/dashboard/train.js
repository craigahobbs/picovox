// The training dashboard's make train model (no DOM) - the stage graph (data/nn/train-stages.json) and a run's
// bookkeeping ($(TRAIN_DIR)/stages.jsonl, the stamps, make's log, util.log) to each stage's state, the forecast (the
// critical path and the ETA), the utilization samples, and the listening samples by model


// A local time from "YYYY-MM-DD HH:MM:SS" (or with a T) - ms
export function parseLocalTime(text) {
    const match = text.match(/^(\d{4})-(\d\d)-(\d\d)[ T](\d\d):(\d\d):(\d\d)/);
    if (match === null) {
        return NaN;
    }
    const [year, month, day, hours, minutes, seconds] = match.slice(1).map(Number);
    return new Date(year, month - 1, day, hours, minutes, seconds).getTime();
}


// The stages in dependency order (needs and after) - [stage], each after the stages it needs or runs after
export function stageOrder(manifest) {
    const byName = new Map(manifest.stages.map((stage) => [stage.stage, stage]));
    const ordered = [];
    const seen = new Set();
    const visit = (stage) => {
        if (!seen.has(stage.stage)) {
            seen.add(stage.stage);
            for (const name of [...stage.needs, ...stage.after || []]) {
                visit(byName.get(name));
            }
            ordered.push(stage);
        }
    };
    manifest.stages.forEach(visit);
    return ordered;
}


// The stages a make target builds - a stage and every stage it needs, or one of the manifest's targets (make train,
// train-check, train-g2p: their stages and every stage those need); an unknown target - every stage
export function targetStages(manifest, target) {
    const byName = new Map(manifest.stages.map((stage) => [stage.stage, stage]));
    const roots = manifest.targets?.[target] ?? (byName.has(target) ? [target] : null);
    if (roots === null) {
        return new Set(byName.keys());
    }
    const included = new Set();
    const visit = (name) => {
        included.add(name);
        byName.get(name).needs.forEach(visit);
    };
    roots.forEach(visit);
    return included;
}


// The latest make target recorded in stages.jsonl ({"event": "target", ...} - make train, train-check, train-g2p) -
// {target, time (ms)}, or null
export function parseRunTarget(text) {
    let latest = null;
    for (const line of text.split('\n')) {
        let record;
        try {
            record = JSON.parse(line);
        } catch {
            continue;
        }
        if (record.event === 'target') {
            latest = {'target': record.target, 'time': record.time * 1000};
        }
    }
    return latest;
}


// Parse stages.jsonl (tools/trainstage.sh) - Map stage -> {start, end, fail, failure} of its latest start (ms)
export function parseStageEvents(text) {
    const stages = new Map();
    for (const line of text.split('\n')) {
        let record;
        try {
            record = JSON.parse(line);
        } catch {
            continue;
        }
        const time = record.time * 1000;
        if (record.event === 'start') {
            stages.set(record.stage, {'start': time, 'end': null, 'fail': null, 'failure': null});
        } else if (stages.has(record.stage) && record.event === 'end') {
            stages.get(record.stage).end = time;
        } else if (record.event === 'fail') {
            const entry = stages.get(record.stage) ?? {'start': null, 'end': null};
            stages.set(record.stage, {...entry, 'fail': time, 'failure': record.signal ?? `exit status ${record.status}`});
        }
    }
    return stages;
}


// Parse make's log (a make train run's output, with the "YYYY-MM-DD HH:MM:SS start:" / "restart:" lines of whoever
// started it) - its segments [{start, target, directory, ended, text}], each a make invocation: its target (from "make
// [-jN] TARGET" in the line), its output's lines up to the next start line, and whether it ended (make left its
// directory). Makes may overlap - another make's start line, its output elsewhere - so make's "Leaving directory" line ends
// the latest segment that entered that directory.
export function parseMakeLog(text) {
    const segments = [];
    for (const line of text.split('\n')) {
        const match = line.match(/^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (?:re)?start: (.*)$/);
        const directory = line.match(/^make(?:\[\d+\])?: (Entering|Leaving) directory [`'"](.*)['"]$/);
        if (match !== null) {
            const make = match[2].match(/make(?: -j ?\d+)? (\S+)/);
            const target = make === null ? 'train' : make[1].replace(/[),;]+$/, '').replace(/^.*\//, '').replace(/\.done$/, '');
            segments.push({'start': parseLocalTime(match[1]), target, 'directory': null, 'ended': false, 'lines': []});
        } else if (segments.length !== 0) {
            const segment = segments[segments.length - 1];
            segment.lines.push(line);
            if (directory !== null && directory[1] === 'Entering' && segment.directory === null) {
                [, , segment.directory] = directory;
            } else if (directory !== null && directory[1] === 'Leaving') {
                const entered = segments.filter((other) => other.directory === directory[2] && !other.ended).pop();
                if (entered) {
                    entered.ended = true;
                }
            }
        }
    }
    return segments.map(({lines, ...segment}) => ({...segment, 'text': lines.join('\n')}));
}


// Parse util.log (tools/trainutil.sh, or an earlier sampler's "HH:MM:SS cpu gpu stages" lines dated from baseTime) -
// [{time, cpu, gpu, mem}] - each header line names the columns
export function parseUtilLog(text, baseTime) {
    const samples = [];
    let columns = ['time', 'cpu_busy%', 'gpu%', 'mem_used%'];
    let dayStart = new Date(baseTime).setHours(0, 0, 0, 0);
    let previous = null;
    for (const line of text.split('\n')) {
        const fields = line.trim().split(/\s+/);
        if (fields[0] === 'time') {
            columns = fields;
            continue;
        }
        const header = columns;
        const indexes = ['cpu_busy%', 'gpu%', 'mem_used%'].map((name) => header.indexOf(name));
        const [cpu, gpu, mem] = indexes.map((index) => (index === -1 ? NaN : Number(fields[index])));
        let time;
        if (fields[0].includes('T')) {
            time = parseLocalTime(fields[0]);
        } else {
            const [hours, minutes, seconds] = fields[0].split(':').map(Number);
            time = dayStart + ((hours * 60 + minutes) * 60 + seconds) * 1000;
            if (previous !== null && time < previous - 12 * 3600 * 1000) {
                // Past midnight
                dayStart += 24 * 3600 * 1000;
                time += 24 * 3600 * 1000;
            }
        }
        if (!Number.isFinite(time) || !Number.isFinite(cpu)) {
            continue;
        }
        previous = time;
        samples.push({time, cpu, gpu, mem});
    }
    return samples;
}


// The mean CPU and GPU utilization of the samples from start to end - {cpu, gpu, count}
export function meanUtilization(samples, start, end) {
    const within = samples.filter((sample) => sample.time >= start && sample.time <= end);
    const mean = (key) => (within.length === 0 ? NaN : within.reduce((sum, sample) => sum + sample[key], 0) / within.length);
    return {'cpu': mean('cpu'), 'gpu': mean('gpu'), 'count': within.length};
}


// Each stage's state - Map stage -> {status, start, end, failure}: status is done, failed, running, queued (waiting for
// the GPU lock), stopped (started, but make isn't running), skipped (not in a run's target), or waiting. From the stage
// events when there are any, else inferred from the stamps (done at their times) and make's log: a make run (segment)
// is open until it leaves its directory or its target stage is done; a stage in an open run's target runs once the
// stages it needs (and runs after) are done; a stage a make reported failed (make's log, or another log in errorTexts)
// failed.
//
// run: {events (parseStageEvents), stamps (Map stage -> ms), segments (parseMakeLog), errorTexts ([text]), alive (make
//       runs - with stage events), target (else the segments'), started (the run's start, ms - earlier unfinished attempts
//       are stale), queued (Set of stages waiting for the GPU lock)}
export function stageStates(manifest, run) {
    const states = new Map();
    const segments = run.segments.map((segment) => (
        {...segment, 'stages': targetStages(manifest, segment.target), 'open': !segment.ended && !run.stamps.has(segment.target)}
    ));
    const open = segments.filter((segment) => segment.open);
    let targets = open.length !== 0 ? open : segments.slice(-1);
    if (run.target) {
        targets = [{'stages': targetStages(manifest, run.target), 'start': null}];
    }
    const included = new Set(targets.flatMap((segment) => [...segment.stages]));
    if (targets.length === 0) {
        targetStages(manifest, manifest.target).forEach((name) => included.add(name));
    }
    const errorTexts = [...segments.slice(-1).map((segment) => segment.text), ...run.errorTexts || []];
    for (const stage of stageOrder(manifest)) {
        const name = stage.stage;
        const deps = [...stage.needs, ...stage.after || []].map((dep) => states.get(dep));
        const depsDone = deps.every((dep) => dep.status === 'done');
        // The latest end of the stage's dependencies (before a time)
        const depsEnd = (before) => Math.max(-Infinity, ...deps.map((dep) => dep.end ?? -Infinity).filter((end) => end <= before));
        // The latest start of a make run building the stage (before a time)
        const runStart = (runs, before) => Math.max(-Infinity, ...runs.
            filter((segment) => segment.stages.has(name) && segment.start <= before).map((segment) => segment.start));
        // (an unfinished attempt from before the run's start is stale - a stage that failed or stopped in an earlier run)
        const recorded = run.events.get(name);
        const event = recorded && run.started && recorded.start < run.started && (recorded.end === null || recorded.end < recorded.start) ?
            undefined : recorded;
        let state;
        if (event && event.end !== null && event.end >= event.start) {
            state = {'status': 'done', 'start': event.start, 'end': event.end};
        } else if (event && event.fail !== null) {
            state = {'status': 'failed', 'start': event.start, 'end': event.fail, 'failure': event.failure};
        } else if (event) {
            const status = run.queued?.has(name) ? 'queued' : 'running';
            state = {'status': run.alive ? status : 'stopped', 'start': event.start, 'end': null};
        } else if (run.stamps.has(name)) {
            const end = run.stamps.get(name);
            const start = Math.max(depsEnd(end), runStart(segments, end));
            state = {'status': 'done', 'start': Number.isFinite(start) ? start : null, end};
        } else if (errorTexts.some((text) => text.includes(`/${name}.done] Error`))) {
            state = {'status': 'failed', 'start': null, 'end': null, 'failure': 'make reported an error'};
        } else if (run.events.size === 0 && open.some((segment) => segment.stages.has(name)) && depsDone) {
            const start = Math.max(depsEnd(Infinity), runStart(open, Infinity));
            state = {'status': 'running', 'start': Number.isFinite(start) ? start : null, 'end': null};
        } else {
            state = {'status': included.has(name) ? 'waiting' : 'skipped', 'start': null, 'end': null};
        }
        states.set(name, state);
    }
    return states;
}


// The forecast - each stage's expected finish and the run's ETA (ms) and critical path. Not-done stages are scheduled in
// dependency order: each starts when the stages it needs (and runs after) finish, a GPU stage also when the GPU is free
// (one at a time), and takes its remaining minutes - remaining(stage, state).
export function forecast(manifest, states, remaining, now) {
    const finishes = new Map();
    const preceding = new Map();
    let gpuFree = now;
    let gpuLast = null;
    for (const stage of stageOrder(manifest)) {
        const name = stage.stage;
        const state = states.get(name);
        const deps = [...stage.needs, ...stage.after || []].filter((dep) => finishes.has(dep));
        const latestDep = deps.reduce((best, dep) => (best === null || finishes.get(dep) > finishes.get(best) ? dep : best), null);
        if (state.status === 'skipped') {
            continue;
        }
        if (state.status === 'done') {
            finishes.set(name, state.end);
            preceding.set(name, latestDep);
            continue;
        }
        let start = Math.max(now, latestDep === null ? now : finishes.get(latestDep));
        let before = latestDep;
        if (stage.on === 'gpu' && !['running', 'stopped'].includes(state.status) && gpuFree > start) {
            start = gpuFree;
            before = gpuLast;
        }
        const finish = start + remaining(stage, state) * 60000;
        if (stage.on === 'gpu') {
            gpuFree = Math.max(gpuFree, finish);
            gpuLast = name;
        }
        finishes.set(name, finish);
        preceding.set(name, before);
    }
    let last = null;
    for (const [name, finish] of finishes) {
        if (last === null || finish > finishes.get(last)) {
            last = name;
        }
    }
    const path = [];
    for (let name = last; name !== null && name !== undefined; name = preceding.get(name)) {
        path.unshift(name);
    }
    return {finishes, 'eta': last === null ? now : finishes.get(last), path};
}


// The listening samples by model - [{key, stage, status, samples}] for each model with a samples stage in the manifest
// (data/nn/train-stages.json's samples - teacher-a, teacher-b, student, polish) or samples in the index
// ($(TRAIN_DIR)/samples/index.json, tools/trainsamples.js), in the order of the training stages they follow (the samples
// key is the training stage's name), the reference (the shipped model - key reference) first. stage is the samples stage
// (null for an index-only model - a run before the samples stages), status its state (done when its samples are in the
// index), and samples its index entries ([] until rendered). A model whose samples stage isn't in the run's make target
// (skipped) and has no samples isn't listed.
export function listenModels(manifest, states, index) {
    const entries = Array.isArray(index) ? index : [];
    const stageOf = new Map(manifest.stages.filter((stage) => stage.samples).map((stage) => [stage.samples, stage.stage]));
    const keys = [...new Set([...stageOf.keys(), ...entries.map((entry) => entry.stage)])];
    const order = stageOrder(manifest).map((stage) => stage.stage);
    const rank = (key) => (key === 'reference' ? -1 : order.indexOf(key) === -1 ? order.length : order.indexOf(key));
    keys.sort((first, second) => rank(first) - rank(second));
    return keys.map((key) => {
        const samples = entries.filter((entry) => entry.stage === key);
        const stage = stageOf.get(key) ?? null;
        const status = samples.length !== 0 ? 'done' : (states.get(stage)?.status ?? 'waiting');
        return {key, stage, status, samples};
    }).filter((model) => model.status !== 'skipped');
}
