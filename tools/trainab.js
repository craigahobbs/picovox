// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// make train-check's blind A/B - the voice lab (build/voicelab - make voicelab VOICELAB_TRAINED) served locally with its
// trials (tools/trainabreport.js): the trained voice against the shipped one, each of the package's voices speaking a
// sentence and a passage, with repeats. The answers are saved as they're given (OUT/answers.json - a rerun resumes);
// when a round is answered, the report and its recommendation (printed, and OUT/report.json), and the server exits. A
// rerun after a finished round starts the next round; the report counts every round. Ctrl-C prints the report so far.
//
//     node tools/trainab.js --lab build/voicelab --out build/train/listen [--summary build/train/distill/summary.json]
//         [--port 8770] [--open false - the browser not opened]

import {abReport, formatReport, makeQuestions} from './trainabreport.js';
import {dirname, extname, join, normalize, resolve} from 'node:path';
import {existsSync, mkdirSync, readFileSync, renameSync, writeFileSync} from 'node:fs';
import {Buffer} from 'node:buffer';
import {createServer} from 'node:http';
import {fileURLToPath} from 'node:url';
import process from 'node:process';
import {spawn} from 'node:child_process';
import {voiceTable} from '../lib/synthesis.js';


// Print a line
const print = (line) => process.stdout.write(`${line}\n`);


// The repository root (tools/ is below it) - the panel texts and the recipe
const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..');


// The command line
const args = {'lab': 'build/voicelab', 'out': 'build/train/listen', 'summary': null, 'port': '8770', 'open': 'true'};
for (let index = 2; index < process.argv.length; index += 2) {
    const name = process.argv[index].replace(/^--/, '');
    if (!(name in args) || index + 1 >= process.argv.length) {
        throw new Error('usage: node tools/trainab.js --lab DIR --out DIR [--summary PATH] [--port PORT] [--open false]');
    }
    args[name] = process.argv[index + 1];
}


// A JSON file's value, or a default when it's missing; written atomically
const readJSON = (path, missing) => (existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) : missing);
function writeJSON(path, value) {
    writeFileSync(`${path}.tmp`, `${JSON.stringify(value, null, 1)}\n`);
    renameSync(`${path}.tmp`, path);
}


// The trials - the earlier rounds', and a new round when they're all answered: each voice's sentence (the panel's
// Harvard sentences) and passage (the recipe's listening samples texts)
mkdirSync(args.out, {'recursive': true});
const questionsPath = join(args.out, 'questions.json');
const answersPath = join(args.out, 'answers.json');
const reportPath = join(args.out, 'report.json');
const questions = readJSON(questionsPath, []);
const answers = readJSON(answersPath, {});
const unanswered = () => questions.filter((question) => !(question.id in answers));
if (unanswered().length === 0) {
    const panelTexts = readJSON(join(repository, 'tools', 'panel-texts.json'), null).texts;
    const recipe = readJSON(join(repository, 'data', 'nn', 'train.json'), null);
    const round = questions.reduce((latest, question) => Math.max(latest, question.round), 0) + 1;
    questions.push(...makeQuestions({
        'voices': voiceTable.map((voice) => voice.key), round,
        'sentences': panelTexts.filter((text) => text.kind === 'harvard').map((text) => text.text),
        'passages': recipe.samples.texts.map((text) => text.text)
    }));
    writeJSON(questionsPath, questions);
}


// The report - printed, and written to OUT/report.json - with the listener panel's scores (the gate's summary)
function report() {
    const result = abReport(questions, answers);
    let panel = null;
    const summary = args.summary === null ? null : readJSON(args.summary, null);
    if (summary !== null) {
        const rows = Object.fromEntries(summary.rows.map((row) => [row.name, row]));
        panel = rows.student && rows[summary.against] ? {'student': rows.student, 'shipped': rows[summary.against]} : null;
    }
    writeJSON(reportPath, result);
    print(formatReport(result, panel).join('\n'));
}


// The server - the voice lab's files, and the A/B's API: the questions, the answers, and an answer's post
const types = {'.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.css': 'text/css', '.svg': 'image/svg+xml'};
const labDir = resolve(args.lab);
const server = createServer((request, response) => {
    const send = (status, body, type = 'application/json') => {
        response.writeHead(status, {'Content-Type': type, 'Cache-Control': 'no-store'});
        response.end(body);
    };
    const url = new URL(request.url, 'http://localhost');
    if (request.method === 'GET' && url.pathname === '/api/questions') {
        send(200, JSON.stringify(questions));
    } else if (request.method === 'GET' && url.pathname === '/api/answers') {
        send(200, JSON.stringify(answers));
    } else if (request.method === 'POST' && url.pathname.startsWith('/api/answers/')) {
        const id = decodeURIComponent(url.pathname.slice('/api/answers/'.length));
        const chunks = [];
        request.on('data', (chunk) => chunks.push(chunk));
        request.on('end', () => {
            if (!questions.some((question) => question.id === id)) {
                send(404, '{}');
                return;
            }
            answers[id] = JSON.parse(Buffer.concat(chunks).toString('utf8'));
            writeJSON(answersPath, answers);
            send(200, '{}');
            if (unanswered().length === 0) {
                report();
                server.close();
                process.exit(0);
            }
        });
    } else if (request.method === 'GET') {
        const path = normalize(join(labDir, url.pathname === '/' ? 'index.html' : decodeURIComponent(url.pathname)));
        if (!path.startsWith(labDir) || !existsSync(path)) {
            send(404, 'Not found', 'text/plain');
            return;
        }
        send(200, readFileSync(path), types[extname(path)] ?? 'application/octet-stream');
    } else {
        send(405, '{}');
    }
});


// Start - the address (and the browser, where one opens), and Ctrl-C's report so far
server.listen(Number(args.port), '127.0.0.1', () => {
    const address = `http://localhost:${args.port}/`;
    const remaining = unanswered().length;
    print(`make train-check: the blind A/B - ${remaining} trials (about ${Math.ceil(remaining / 4)} minutes) at ${address} ` +
        '- the report prints here when you finish (Ctrl-C stops; a rerun resumes)');
    const opener = {'darwin': 'open', 'linux': 'xdg-open'}[process.platform];
    if (opener && args.open !== 'false') {
        spawn(opener, [address], {'stdio': 'ignore', 'detached': true}).on('error', () => null).unref();
    }
});
process.on('SIGINT', () => {
    print(`\nmake train-check: stopped - ${unanswered().length} trials unanswered (make train-check resumes)`);
    report();
    process.exit(130);
});
