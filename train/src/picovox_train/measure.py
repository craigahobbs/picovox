# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The readings' measurement - the readings the training speech's voices used for its heteronym tokens (tts.py's corpus;
readings.py's words and classes), measured with our own tools: the CTC phoneme recognizer's vote (ctc.py) where the
readings differ in a full vowel's quality or a consonant, and the vowel's formants (Praat) where the recognizer can't
tell them apart (a reduced vowel against a full one, the vowels it merges). The G2P's heteronym evidence
(heteronyms.py, contextlabels.py) is these readings, and the voice's tokens (tts.py) take the measured reading where
the CTC vote differs from the front end's.

    PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.measure    # <corpus>/readings/ (GPU)

The measurement of a token (a measured word in a clip, by the clip's alignment): its class (readings.pair_class)
chooses the method - strong: each reading's CTC likelihood (its first variant, with the neighboring words' phones) on
the word's segment, a vote when one is VOTE_MARGIN more likely; voicing: the same with S and Z each only its own token;
weak and merger: the differing vowel's F1 and F2 (its middle third, Lobanov-normalized per voice), nearest of the
readings' vowels' references (pooled per voice over REFERENCE_CLIPS clips' other vowels) by FORMANT_MARGIN; stress:
none. The controls - words of one pronunciation with a phone swapped (a final S or Z, the stressed vowel for a partner
the recognizer hears apart, a clear vowel for its formant neighbor) - are each method's false votes: a method's
readings are used when they're at most MAX_CONTROL_ERROR of MIN_CONTROL_VOTES or more. The outputs, in
<corpus>/readings/: words.json (the measured words, their classes, and the measurement's key), readings.jsonl (each
clip's measured tokens), and summary.json (the controls, the classes used, the voices' reference vowels).
"""

import argparse
import collections
import hashlib
import json
import os
import sys
import time
import warnings

from . import RECIPE_PATH, read_json
from .readings import CLOSED_CLASS, VOWELS, base, difference, measured_words, read_variants, reading_of, token_words, unstressed, word_class


# The context of a word - its neighbors' phones when at most NEIGHBOR_SECONDS long, else PAD seconds of the audio; the
# controls - CONTROL_WORDS words a method, at most CONTROL_TOKENS tokens a word; the formants' references -
# REFERENCE_CLIPS clips a voice (their unmeasured words' vowels of at least MIN_VOWEL_SECONDS), and a vote's margin
# (Lobanov units); a method's readings are used when its controls' false votes are at most MAX_CONTROL_ERROR of
# MIN_CONTROL_VOTES or more
NEIGHBOR_SECONDS = 1.0
PAD = 0.12
CONTROL_WORDS = 150
CONTROL_TOKENS = 6
REFERENCE_CLIPS = 300
MIN_VOWEL_SECONDS = 0.05
FORMANT_MARGIN = 0.25
MAX_CONTROL_ERROR = 0.03
MIN_CONTROL_VOTES = 100

# The method of each class
METHODS = {'strong': 'ctc', 'voicing': 'voicing', 'weak': 'formant', 'merger': 'formant'}

# The controls' swaps - a clear stressed vowel for a partner the recognizer hears apart (ctc), and for its formant
# neighbor (formant)
CTC_PARTNERS = {'IY': 'EH', 'EH': 'IY', 'IH': 'AY', 'AY': 'IH', 'AE': 'EY', 'EY': 'AE', 'OW': 'AW', 'AW': 'OW', 'UW': 'OW'}
FORMANT_PARTNERS = {'IH': 'IY', 'IY': 'IH', 'EH': 'AE', 'AE': 'EH', 'UH': 'UW', 'UW': 'UH', 'AA': 'AH', 'AH': 'AA'}


# Print a progress line to stderr
def log(message):
    print(message, file=sys.stderr, flush=True)


# The measurement's key - a hash of the measured words and the method's settings (another key measures again)
def measure_key(words):
    from . import ctc, tts  # pylint: disable=import-outside-toplevel
    basis = [words, ctc.VOTE_MARGIN, NEIGHBOR_SECONDS, PAD, CONTROL_WORDS, CONTROL_TOKENS, REFERENCE_CLIPS, MIN_VOWEL_SECONDS,
             FORMANT_MARGIN, CTC_PARTNERS, FORMANT_PARTNERS, tts.aligner_key()]
    return hashlib.sha256(json.dumps(basis, sort_keys=True).encode('utf-8')).hexdigest()[:12]


# A clip's words with their times - [(phones, start, end, labels)] from its alignment record (its phones' labels in
# order, the edge pauses aside; a word without phones, (phones, None, None, []))
def word_times(record):
    labels = [label for label in record['labels'] if label[0] != 'pau']
    words, index = [], 0
    for phones in token_words(record['tokens']):
        count = len(phones.split())
        if count == 0:
            words.append((phones, None, None, []))
        else:
            words.append((phones, labels[index][1], labels[index + count - 1][2], labels[index:index + count]))
            index += count
    return words


# A word's segment and context - (start, end, phones before, phones after)
def word_context(words, position, seconds):
    _, start, end, _ = words[position]
    before, after = [], []
    crop_start, crop_end = max(0.0, start - PAD), min(seconds, end + PAD)
    previous = words[position - 1] if position > 0 else None
    following = words[position + 1] if position + 1 < len(words) else None
    if previous is not None and previous[1] is not None and previous[2] - previous[1] <= NEIGHBOR_SECONDS:
        crop_start, before = previous[1], previous[0].split()
    if following is not None and following[1] is not None and following[2] - following[1] <= NEIGHBOR_SECONDS:
        crop_end, after = following[2], following[0].split()
    return crop_start, crop_end, before, after


# The S and Z positions of a variant (the voicing method's strict positions)
def voicing_positions(variant):
    return {index for index, phone in enumerate(variant.split()) if base(phone) in ('S', 'Z')}


# A clip's measurement tasks - its measured words' tokens ({position, word, class, frontEnd, method, segment, phones,
# labels, candidates}) and at most two control tokens ({control: True, ...}); words, the measured words' readings and
# classes; controls, {word: (method, candidates)}; names, the clip's words (the front end's)
def clip_tasks(record, words, classes, controls, names):
    timed = word_times(record)
    tasks = []
    for position, (phones, start, _, labels) in enumerate(timed):
        if start is None or position >= len(names):
            continue
        name = names[position]
        segment = word_context(timed, position, record['seconds'])
        if name in words and classes[name] != 'stress':
            readings = words[name]
            tasks.append({'position': position, 'word': name, 'class': classes[name], 'frontEnd': reading_of(phones, readings),
                          'method': METHODS[classes[name]], 'segment': segment, 'phones': phones, 'labels': labels,
                          'candidates': [reading[0] for reading in readings]})
        elif name in controls and controls[name][1][0] == phones and sum(bool(task.get('control')) for task in tasks) < 2:
            method, candidates = controls[name]
            tasks.append({'position': position, 'word': name, 'control': True, 'method': method, 'segment': segment,
                          'phones': phones, 'labels': labels, 'candidates': candidates})
    return tasks


# The control words - {word: (method, [the word's pronunciation, the swapped one])}: the corpus's words of one
# pronunciation (not measured, not closed-class, 4+ letters, CONTROL_TOKENS+ tokens), by a hash, CONTROL_WORDS a method
def control_words(counts, variants, measured):
    choices = {'ctc': [], 'voicing': [], 'formant': []}
    for word in sorted(counts, key=lambda item: hashlib.sha1(item.encode('utf-8')).hexdigest()):
        if word in measured or word in CLOSED_CLASS or len(word) < 4 or len(variants.get(word, [])) != 1 or counts[word] < CONTROL_TOKENS:
            continue
        phones = variants[word][0].split()
        stressed = next((index for index, phone in enumerate(phones) if phone[-1:] == '1'), None)
        vowel = base(phones[stressed]) if stressed is not None else None
        before_r = stressed is not None and stressed + 1 < len(phones) and base(phones[stressed + 1]) == 'R'
        if base(phones[-1]) in ('S', 'Z'):
            method, swapped = 'voicing', phones[:-1] + ['Z' if base(phones[-1]) == 'S' else 'S']
        elif vowel in CTC_PARTNERS and not before_r:
            method, swapped = 'ctc', phones[:stressed] + [CTC_PARTNERS[vowel] + '1'] + phones[stressed + 1:]
        elif vowel in FORMANT_PARTNERS and not before_r:
            method, swapped = 'formant', phones[:stressed] + [FORMANT_PARTNERS[vowel] + '1'] + phones[stressed + 1:]
        else:
            continue
        if len(choices[method]) < CONTROL_WORDS:
            choices[method].append((word, ' '.join(swapped)))
    return {word: (method, [variants[word][0], swapped]) for method, items in choices.items() for word, swapped in items}


# The CTC votes of tasks [(clip path, task)] (in place - each task's scores and vote)
def ctc_votes(model, vocab, device, items):
    from . import ctc  # pylint: disable=import-outside-toplevel
    segments = [ctc.read_segment(path, task['segment'][0], task['segment'][1]) for path, task in items]
    posteriors = ctc.ctc_log_posteriors(model, device, segments)
    for (_, task), log_posteriors in zip(items, posteriors):
        _, _, before, after = task['segment']
        scores = []
        for candidate in task['candidates']:
            voicing = voicing_positions(candidate) if task['method'] == 'voicing' else ()
            scores.append(ctc.candidate_log_likelihood(log_posteriors, before, candidate.split(), after, vocab, voicing))
        task['scores'] = [round(score, 3) for score in scores]
        task['vote'] = ctc.vote(scores)


# A clip's formants (a worker task: (path, sex, ceilings, [(start, end)])) - [[F1, F2] or None], each interval's
# middle third's median
def clip_formants(task):
    import numpy  # pylint: disable=import-outside-toplevel
    import parselmouth  # pylint: disable=import-outside-toplevel, import-error
    path, sex, ceilings, intervals = task
    formant = parselmouth.Sound(path).to_formant_burg(time_step=0.005, max_number_of_formants=5, maximum_formant=ceilings[sex])
    values = []
    for start, end in intervals:
        times = numpy.linspace(start + (end - start) / 3, end - (end - start) / 3, 5)
        # (an interval without a defined formant is None - numpy's all-NaN median warns)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            first = numpy.nanmedian([formant.get_value_at_time(1, moment) for moment in times])
            second = numpy.nanmedian([formant.get_value_at_time(2, moment) for moment in times])
        values.append([round(float(first), 1), round(float(second), 1)] if numpy.isfinite(first) and numpy.isfinite(second) else None)
    return values


# A vowel's reference key - its base, or AH0 (schwa) when reduced
def vowel_key(phone):
    return 'AH0' if phone[-1:] == '0' and base(phone) in ('AH', 'IH', 'ER') else base(phone)


# The differing vowel of a formant task - (its index in the token's phones, {candidate index: the candidate's vowel key})
# or None: the first vowel the other candidates differ from the token's phones by
def formant_target(task):
    phones = task['phones'].split()
    found, vowels = None, {}
    own = None
    for index, candidate in enumerate(task['candidates']):
        if unstressed(candidate) == unstressed(task['phones']):
            own = index
            continue
        differing = difference(phones, candidate.split())
        if differing is None or differing[2] != 'vowel' or found not in (None, differing[0]):
            continue
        found = differing[0]
        vowels[index] = vowel_key(candidate.split()[differing[1]])
    if found is None or own is None:
        return None
    vowels[own] = vowel_key(phones[found])
    return found, vowels


# A formant task's vote - the candidate whose vowel's reference is nearest the token's normalized formants by
# FORMANT_MARGIN, else None
def formant_vote(task, norm, references):
    import numpy  # pylint: disable=import-outside-toplevel
    if task.get('target') is None or task.get('formants') is None or norm is None:
        return None
    normalized = (numpy.array(task['formants']) - norm[0]) / norm[1]
    distances = {index: float(numpy.linalg.norm(normalized - references[vowel]))
                 for index, vowel in task['target'][1].items() if vowel in references}
    if len(set(task['target'][1].values())) < 2 or len(distances) < 2:
        return None
    order = sorted(distances, key=distances.get)
    return order[0] if distances[order[1]] - distances[order[0]] >= FORMANT_MARGIN else None


# The formants of the tasks (in place - each formant task's formants) and each voice's normalization and reference
# vowels - {voice: ((mean, std) or None, {vowel: mean})}
def measure_formants(plan, clips, records, words, names, directory, settings, jobs):
    import multiprocessing  # pylint: disable=import-outside-toplevel
    import numpy  # pylint: disable=import-outside-toplevel
    from . import tts  # pylint: disable=import-outside-toplevel
    intervals = collections.defaultdict(list)
    for clip, task in plan:
        if task['method'] == 'formant':
            task['target'] = formant_target(task)
            if task['target'] is not None:
                label = task['labels'][task['target'][0]]
                intervals[clip['id']].append((task, label[1], label[2]))

    # (each voice's reference clips' unmeasured words' vowels)
    by_voice = collections.defaultdict(list)
    for clip in clips:
        by_voice[clip['key']].append(clip)
    for voice_clips in by_voice.values():
        for clip in sorted(voice_clips, key=lambda item: item['id'])[:REFERENCE_CLIPS]:
            for position, (_, _, _, labels) in enumerate(word_times(records[clip['id']])):
                if position < len(names[clip['id']]) and names[clip['id']][position] not in words:
                    for label in labels:
                        if base(label[0]) in VOWELS and label[2] - label[1] >= MIN_VOWEL_SECONDS:
                            intervals[clip['id']].append(((clip['key'], vowel_key(label[0])), label[1], label[2]))
    by_id = {clip['id']: clip for clip in clips}
    order = sorted(intervals)
    ceilings = settings['analysis']['formantCeilings']
    tasks = [(tts.wav_path(directory, by_id[clip_id]), by_id[clip_id]['sex'], ceilings,
              [(start, end) for _, start, end in intervals[clip_id]]) for clip_id in order]
    references = collections.defaultdict(list)
    with multiprocessing.Pool(jobs) as pool:
        for clip_id, values in zip(order, pool.imap(clip_formants, tasks, chunksize=8)):
            for (owner, _, _), value in zip(intervals[clip_id], values):
                if isinstance(owner, tuple):
                    if value is not None:
                        references[owner].append(value)
                else:
                    owner['formants'] = value

    # (Lobanov normalization per voice, and its pooled reference vowels)
    result = {}
    for voice in by_voice:
        values = numpy.array([value for (owner, _), items in references.items() if owner == voice for value in items])
        if len(values) < 30:
            result[voice] = (None, {})
            continue
        mean, std = values.mean(0), values.std(0)
        result[voice] = ((mean, std), {vowel: ((numpy.array(items) - mean) / std).mean(0)
                                       for (owner, vowel), items in references.items() if owner == voice and len(items) >= 10})
    return result


def measure_command(args):
    """The readings of the corpus's heteronym tokens - <corpus>/readings/ (words.json, readings.jsonl, summary.json)"""
    from . import ctc, tts  # pylint: disable=import-outside-toplevel
    start_time = time.time()
    settings = tts.load_settings(args.recipe)
    directory = tts.corpus_dir(settings)
    out = os.path.join(directory, 'readings')
    words = measured_words()
    classes = {word: word_class(readings) for word, readings in words.items()}
    key = measure_key(words)
    if not args.limit and os.path.exists(os.path.join(out, 'summary.json')) and read_json(os.path.join(out, 'summary.json'))['key'] == key:
        log(f'measure: {out} is current')
        return 0
    records = tts.read_alignments(directory)
    clips = [clip for clip in tts.selection(settings, args.jobs, args.recipe)
             if clip['id'] in records and not tts.clip_checks(records[clip['id']], settings['checks'])[0]]
    if args.limit:
        clips = clips[::max(1, len(clips) // args.limit)][:args.limit]
    names = {clip_id: front['words'] for clip_id, front in tts.clip_fronts(clips, {}, args.jobs).items()}
    counts = collections.Counter(name for clip_names in names.values() for name in clip_names)
    controls = control_words(counts, read_variants(), words)

    # The tokens - each clip's measured words' and controls'
    plan = [(clip, task) for clip in clips for task in clip_tasks(records[clip['id']], words, classes, controls, names[clip['id']])]
    per_control = collections.Counter()
    kept = []
    for clip, task in plan:
        if task.get('control'):
            per_control[task['word']] += 1
            if per_control[task['word']] > CONTROL_TOKENS:
                continue
        kept.append((clip, task))
    plan = kept
    log(f'measure: {len(clips)} clips, {sum(not task.get("control") for _, task in plan)} measured tokens, '
        f'{sum(bool(task.get("control")) for _, task in plan)} control tokens')

    # The CTC votes, and the formant votes
    model, vocab, device = ctc.load_ctc(args.device)
    voted = [(tts.wav_path(directory, clip), task) for clip, task in plan if task['method'] in ('ctc', 'voicing')]
    for offset in range(0, len(voted), args.batch):
        ctc_votes(model, vocab, device, voted[offset:offset + args.batch])
        log(f'  ctc {min(len(voted), offset + args.batch)}/{len(voted)} tokens, {time.time() - start_time:.0f} s')
    voices = measure_formants(plan, clips, records, words, names, directory, settings, args.jobs)
    for clip, task in plan:
        if task['method'] == 'formant':
            task['vote'] = formant_vote(task, *voices[clip['key']])

    # The controls - each method's false votes, and the classes whose readings are used
    control_results = {}
    for method in ('ctc', 'voicing', 'formant'):
        votes = [task['vote'] for _, task in plan if task.get('control') and task['method'] == method and task['vote'] is not None]
        false = sum(vote != 0 for vote in votes)
        control_results[method] = {'votes': len(votes), 'false': false, 'error': round(false / max(1, len(votes)), 4),
                                   'used': len(votes) >= MIN_CONTROL_VOTES and false <= MAX_CONTROL_ERROR * len(votes)}
    used = {name: control_results[method]['used'] for name, method in METHODS.items()}

    # The readings - each clip's measured tokens (a token's reading its vote, when its class is used)
    os.makedirs(out, exist_ok=True)
    by_clip = collections.defaultdict(list)
    for clip, task in plan:
        if not task.get('control'):
            by_clip[clip['id']].append(task)
    with open(os.path.join(out, 'readings.jsonl'), 'w', encoding='utf-8') as file:
        for clip in clips:
            tokens = [{'position': task['position'], 'word': task['word'], 'class': task['class'], 'frontEnd': task['frontEnd'],
                       'vote': task['vote'], 'reading': task['vote'] if used[task['class']] else None,
                       **({'scores': task['scores']} if 'scores' in task else {'formants': task.get('formants')})}
                      for task in by_clip.get(clip['id'], [])]
            if tokens:
                file.write(json.dumps({'id': clip['id'], 'key': clip['key'], 'voice': clip['voice'], 'sex': clip['sex'],
                                       'book': clip['book'], 'text': clip['text'], 'split': clip['split'], 'tokens': tokens}) + '\n')
    with open(os.path.join(out, 'words.json'), 'w', encoding='utf-8') as file:
        json.dump({'key': key, 'words': words, 'classes': classes}, file)
    measured = [task for _, task in plan if not task.get('control')]
    summary = {'key': None if args.limit else key, 'corpus': os.path.basename(directory), 'clips': len(clips), 'tokens': len(measured),
               'byClass': dict(collections.Counter(task['class'] for task in measured)),
               'votes': dict(collections.Counter(task['class'] for task in measured if task['vote'] is not None)),
               'controls': control_results, 'classes': used,
               'references': {voice: {vowel: [round(float(value), 3) for value in mean] for vowel, mean in vowels.items()}
                              for voice, (_, vowels) in voices.items()}}
    with open(os.path.join(out, 'summary.json'), 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=1)
    log(f'measure: {len(measured)} tokens, votes {summary["votes"]}, in {time.time() - start_time:.0f} s; controls '
        f'{ {method: (result["false"], result["votes"]) for method, result in control_results.items()} }; classes used {used}')
    return 0


# The command line
def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m picovox_train.measure',
                                     description='The training speech\'s heteronym readings, measured')
    parser.add_argument('--recipe', default=RECIPE_PATH, help='the recipe (default train/data/train.json)')
    parser.add_argument('--batch', type=int, default=256, help='segments at once')
    parser.add_argument('--device', default=None, help='torch device (default mps if available)')
    parser.add_argument('--limit', type=int, help='a sample of this many clips (a quick test - not kept as current)')
    parser.add_argument('-j', '--jobs', type=int, default=None, help='worker processes (default: all cores)')
    return measure_command(parser.parse_args(argv))

if __name__ == '__main__':
    sys.exit(main())
