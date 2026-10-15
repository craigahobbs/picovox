// The training dashboard's monitor - every second, writes build/nn/activity.json with the steps that write no log:
// - processes: the Python and Node processes working in this repository (each with its multiprocessing workers folded
//   in), running now or before
// - agents: the workflow's agents (pipeline.json's workflow, else the newest workflow under ~/.claude/projects) - each
//   one's phase, whether it finished, and its latest action
//
// node tools/dashboard/monitor.js

import {
    closeSync, existsSync, mkdirSync, openSync, readFileSync, readSync, readdirSync, renameSync, statSync, writeFileSync
} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import {Buffer} from 'node:buffer';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {homedir} from 'node:os';
import process from 'node:process';


// The repository, the files, the refresh period (ms), and how long an ended process stays in the history (ms)
const repository = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const activityPath = join(repository, 'build/nn/activity.json');
const pipelinePath = join(repository, 'tools/dashboard/pipeline.json');
const projectsPath = join(homedir(), '.claude/projects');
const periodMs = 1000;
const keepMs = 24 * 60 * 60 * 1000;
const minimumMs = 10000;


// Interpreters whose processes may be pipeline steps, and commands that never are
const interpreterPattern = /(?:^|\/)(?:Python|python[\d.]*|node)$/;
const ignorePattern = /http\.server|tools\/dashboard\/|node_modules|eslint|Google Chrome/;


// Parse ps's elapsed time ([[dd-]hh:]mm:ss) - seconds
export function parseElapsed(text) {
    const [days, clock] = text.includes('-') ? text.split('-') : ['0', text];
    const parts = clock.split(':').map(Number);
    while (parts.length < 3) {
        parts.unshift(0);
    }
    const [hours, minutes, seconds] = parts;
    return Number(days) * 86400 + hours * 3600 + minutes * 60 + seconds;
}


// A process's step label from its command - the module or script and its subcommand (e.g. "nn.multi build")
export function commandLabel(command) {
    const tokens = command.replace(/\\012/g, ' ').trim().split(/\s+/);
    let index = 1;
    while (index < tokens.length && /^-[A-Za-z]$/.test(tokens[index]) && !['-m', '-c'].includes(tokens[index])) {
        index++;
    }
    let name;
    if (tokens[index] === '-m') {
        name = tokens[index + 1] || 'python -m';
        index += 2;
    } else if (tokens[index] === '-c') {
        return `python -c ${tokens.slice(index + 1).join(' ').slice(0, 40)}`;
    } else {
        name = (tokens[index] || tokens[0]).replace(/^\.\//, '');
        if (name.startsWith('/')) {
            name = name.slice(name.lastIndexOf('/') + 1);
        }
        index += 1;
    }
    const subcommand = tokens.slice(index).find((token) => !token.startsWith('-') && token.length <= 40);
    return subcommand ? `${name} ${subcommand}` : name;
}


// The processes - [{pid, ppid, elapsed, cpu, command}]
function listProcesses() {
    const options = {'encoding': 'utf8', 'maxBuffer': 16 * 1024 * 1024};
    const output = execFileSync('ps', ['-axo', 'pid=,ppid=,etime=,%cpu=,command='], options);
    return output.split('\n').filter((line) => line.trim() !== '').map((line) => {
        const [pid, ppid, elapsed, cpu, ...command] = line.trim().split(/\s+/);
        return {
            'pid': Number(pid), 'ppid': Number(ppid), 'elapsed': parseElapsed(elapsed), 'cpu': Number(cpu), 'command': command.join(' ')
        };
    });
}


// A process's working directory (cached - it doesn't change)
const cwdCache = new Map();
function processCwd(pid) {
    if (!cwdCache.has(pid)) {
        let cwd = '';
        try {
            const output = execFileSync('lsof', ['-a', '-d', 'cwd', '-Fn', '-p', String(pid)], {'encoding': 'utf8'});
            cwd = (output.split('\n').find((line) => line.startsWith('n')) || 'n').slice(1);
        } catch {
            // Gone, or not ours
        }
        cwdCache.set(pid, cwd);
    }
    return cwdCache.get(pid);
}


// The running steps - [{label, command, started, cpu, workers}], each the top-most interpreter process working in the
// repository, with its descendant interpreter processes (workers) folded in
function runningSteps(now) {
    const processes = listProcesses();
    const byPid = new Map(processes.map((entry) => [entry.pid, entry]));
    const candidates = new Set(processes.filter((entry) => {
        const [executable] = entry.command.split(/\s+/);
        const cwd = processCwd(entry.pid);
        const inRepository = cwd === repository || cwd.startsWith(`${repository}/`);
        return interpreterPattern.test(executable) && !ignorePattern.test(entry.command) && inRepository;
    }).map((entry) => entry.pid));
    const steps = new Map();
    for (const pid of candidates) {
        // The top-most candidate ancestor owns this process
        let owner = pid;
        for (let parent = byPid.get(pid).ppid; byPid.has(parent); parent = byPid.get(parent).ppid) {
            if (candidates.has(parent)) {
                owner = parent;
            }
        }
        const ownerEntry = byPid.get(owner);
        if (!steps.has(owner)) {
            steps.set(owner, {'label': commandLabel(ownerEntry.command), 'command': ownerEntry.command.slice(0, 300),
                'started': now - ownerEntry.elapsed * 1000, 'cpu': 0, 'workers': 0});
        }
        const step = steps.get(owner);
        step.cpu += byPid.get(pid).cpu;
        if (pid !== owner) {
            step.workers += 1;
        }
    }
    return [...steps.values()];
}


// Update the process history with the running steps - a step is its label and start (a restart is a new entry), and
// processes running less than 10 seconds (quick checks) aren't steps
export function updateHistory(history, running, now) {
    const keyOf = (step) => `${step.label} ${Math.round(step.started / 10000)}`;
    const entries = new Map(history.map((entry) => [keyOf(entry), {...entry, 'running': false}]));
    for (const step of running.filter((item) => now - item.started >= minimumMs)) {
        const entry = entries.get(keyOf(step)) || {...step};
        Object.assign(entry, {'cpu': Math.round(step.cpu), 'workers': step.workers, 'running': true, 'lastSeen': now});
        entries.set(keyOf(step), entry);
    }
    const kept = (entry) => entry.running || (now - entry.lastSeen < keepMs && entry.lastSeen - entry.started >= minimumMs);
    return [...entries.values()].filter(kept).
        sort((first, second) => first.started - second.started);
}


// The workflow's directory - pipeline.json's workflow (a run id), else the newest - rediscovered every 30 seconds
let workflowDir = null;
let workflowFound = 0;
function findWorkflowDir(now) {
    if (now - workflowFound < 30000) {
        return workflowDir;
    }
    workflowFound = now;
    let wanted = null;
    try {
        wanted = JSON.parse(readFileSync(pipelinePath, 'utf8')).workflow || null;
    } catch {
        // No pipeline - the newest workflow
    }
    let newest = null;
    const list = (path) => {
        try {
            return readdirSync(path);
        } catch {
            return [];
        }
    };
    for (const project of list(projectsPath)) {
        for (const session of list(join(projectsPath, project))) {
            const workflows = join(projectsPath, project, session, 'subagents/workflows');
            for (const workflow of list(workflows)) {
                const journal = join(workflows, workflow, 'journal.jsonl');
                if (existsSync(journal) && (wanted === null || workflow === wanted)) {
                    const modified = statSync(journal).mtimeMs;
                    if (newest === null || modified > newest.modified) {
                        newest = {'path': join(workflows, workflow), modified};
                    }
                }
            }
        }
    }
    workflowDir = newest?.path ?? null;
    return workflowDir;
}


// Read up to the last bytes of a file
function readTail(path, bytes) {
    const {size} = statSync(path);
    const length = Math.min(size, bytes);
    const buffer = Buffer.alloc(length);
    const descriptor = openSync(path, 'r');
    try {
        readSync(descriptor, buffer, 0, length, size - length);
    } finally {
        closeSync(descriptor);
    }
    return buffer.toString('utf8');
}


// A transcript's latest action - {action, time} - from its last tool call with a description (or command or file)
export function latestAction(text) {
    const lines = text.split('\n');
    for (let index = lines.length - 1; index >= 0; index--) {
        let record;
        try {
            record = JSON.parse(lines[index]);
        } catch {
            continue;
        }
        const content = record.message?.role === 'assistant' && Array.isArray(record.message.content) ? record.message.content : [];
        for (const item of [...content].reverse()) {
            if (item.type === 'tool_use') {
                const input = item.input || {};
                const action = input.description || input.command || input.file_path || input.pattern || item.name;
                const [firstLine] = String(action).split('\n');
                return {'action': `${item.name}: ${firstLine.slice(0, 160)}`, 'time': Date.parse(record.timestamp) || null};
            }
        }
    }
    return null;
}


// The workflow's agents - [{id, label, phase, status, started, updated, action, actionTime}]
const transcriptCache = new Map();
function workflowAgents(now) {
    const directory = findWorkflowDir(now);
    if (directory === null) {
        return [];
    }
    const agents = new Map();
    let journal;
    try {
        journal = readFileSync(join(directory, 'journal.jsonl'), 'utf8');
    } catch {
        return [];
    }
    for (const line of journal.split('\n')) {
        let record;
        try {
            record = JSON.parse(line);
        } catch {
            continue;
        }
        if (record.type === 'started' && record.agentId) {
            const agent = {'id': record.agentId, 'label': record.label || record.agentId, 'phase': record.phase || '', 'status': 'running'};
            agents.set(record.agentId, agent);
        } else if (record.type === 'result' && agents.has(record.agentId)) {
            agents.get(record.agentId).status = 'done';
        }
    }
    for (const agent of agents.values()) {
        const transcript = join(directory, `agent-${agent.id}.jsonl`);
        if (!existsSync(transcript)) {
            continue;
        }
        const {mtimeMs, birthtimeMs} = statSync(transcript);
        let cached = transcriptCache.get(transcript);
        if (!cached || cached.modified !== mtimeMs) {
            cached = {'modified': mtimeMs, 'latest': latestAction(readTail(transcript, 256 * 1024))};
            transcriptCache.set(transcript, cached);
        }
        Object.assign(agent, {'started': birthtimeMs, 'updated': mtimeMs, 'action': cached.latest?.action ?? '',
            'actionTime': cached.latest?.time ?? mtimeMs});
    }
    return [...agents.values()];
}


// Monitor - write the activity file every second
function main() {
    mkdirSync(dirname(activityPath), {'recursive': true});
    let history = [];
    try {
        history = JSON.parse(readFileSync(activityPath, 'utf8')).steps || [];
    } catch {
        // No history yet
    }
    const tick = () => {
        const now = Date.now();
        try {
            history = updateHistory(history, runningSteps(now), now);
            const activity = {'updated': now, 'steps': history, 'agents': workflowAgents(now)};
            const temporary = `${activityPath}.tmp`;
            writeFileSync(temporary, `${JSON.stringify(activity, null, 1)}\n`);
            renameSync(temporary, activityPath);
        } catch (error) {
            process.stderr.write(`monitor: ${error.message}\n`);
        }
        setTimeout(tick, periodMs);
    };
    process.stdout.write(`Monitoring the pipeline's processes and agents into ${activityPath}\n`);
    tick();
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    main();
}
