'''
This script takes as input the LSTM or RNN weights found by Train.py
and evaluates:
 - next activity prediction
 - time until next activity

Log reading is done via pm4py (CSV event log).

Original idea: Niek Tax / Ilya Verenich
Python 3 + pm4py adaptation: Lyan Eduardo Sakuno Rodrigues
'''

from __future__ import division

from keras.models import load_model
import csv
import copy
import numpy as np
import distance
from jellyfish._jellyfish import damerau_levenshtein_distance
from sklearn import metrics
from math import sqrt
from datetime import timedelta
import matplotlib.pyplot as plt  # mantido, caso queira usar
from collections import Counter
import os

from pm4py.objects.log.importer.csv import factory as csv_importer

# --------------------------------------------------------------------
# Configurações principais
# --------------------------------------------------------------------

eventlog = "helpdesk.csv"
log_path = os.path.join("data", eventlog)
ascii_offset = 161

# Caminho do modelo treinado (gerado pelo Train.py)
MODEL_PATH = "output_files/models/model_89-1.50.h5"

print(f"Lendo log com pm4py: {log_path}")

parameters = {
    "case_id": "CaseID",
    "activity_key": "ActivityID",
    "timestamp_key": "CompleteTimestamp"
}
log = csv_importer.apply(log_path, parameters=parameters)

# --------------------------------------------------------------------
# Primeira passagem: construir sequências e tempos (para divisores)
# --------------------------------------------------------------------

lines = []
caseids = []
timeseqs = []   # tempos entre eventos consecutivos (segundos)
timeseqs2 = []  # tempos desde o início do caso (segundos)

for trace in log:
    case_id = trace.attributes.get("concept:name", None)
    if case_id is None and len(trace) > 0:
        case_id = trace[0].get("CaseID", None)
    caseids.append(case_id)

    line = ''
    times = []
    times2 = []

    casestarttime = None
    lasteventtime = None

    for event in trace:
        ts = event["CompleteTimestamp"]
        act_id = int(event["ActivityID"])

        if casestarttime is None:
            casestarttime = ts
            lasteventtime = ts

        line += chr(act_id + ascii_offset)

        timesincelastevent = ts - lasteventtime
        timesincecasestart = ts - casestarttime

        timediff = int(timesincelastevent.total_seconds())
        timediff2 = int(timesincecasestart.total_seconds())

        times.append(timediff)
        times2.append(timediff2)

        lasteventtime = ts

    lines.append(line)
    timeseqs.append(times)
    timeseqs2.append(times2)

numlines = len(lines)
print(f"Número de casos (traces): {numlines}")

# --------------------------------------------------------------------
# Divisores (normalização de tempos)
# --------------------------------------------------------------------

divisor = np.mean([item for sublist in timeseqs for item in sublist])
print('divisor: {}'.format(divisor))

divisor2 = np.mean([item for sublist in timeseqs2 for item in sublist])
print('divisor2: {}'.format(divisor2))

# --------------------------------------------------------------------
# Folds (como no original)
# --------------------------------------------------------------------

elems_per_fold = int(round(numlines / 3))

fold1 = lines[:elems_per_fold]
fold1_c = caseids[:elems_per_fold]
fold1_t = timeseqs[:elems_per_fold]
fold1_t2 = timeseqs2[:elems_per_fold]

fold2 = lines[elems_per_fold:2 * elems_per_fold]
fold2_c = caseids[elems_per_fold:2 * elems_per_fold]
fold2_t = timeseqs[elems_per_fold:2 * elems_per_fold]
fold2_t2 = timeseqs2[elems_per_fold:2 * elems_per_fold]

fold3 = lines[2 * elems_per_fold:]
fold3_c = caseids[2 * elems_per_fold:]
fold3_t = timeseqs[2 * elems_per_fold:]
fold3_t2 = timeseqs2[2 * elems_per_fold:]

# folds 1 + 2 usados para vocabulário e maxlen
lines_train = fold1 + fold2
lines_train_t = fold1_t + fold2_t
lines_train_t2 = fold1_t2 + fold2_t2

# --------------------------------------------------------------------
# Construção do vocabulário de caracteres
# --------------------------------------------------------------------

step = 1
sentences = []
softness = 0
next_chars = []

lines_train = [x + '!' for x in lines_train]
maxlen = max(len(x) for x in lines_train)

chars = [set(x) for x in lines_train]
chars = list(set().union(*chars))
chars.sort()
target_chars = copy.copy(chars)
chars.remove('!')

print('total chars: {}, target chars: {}'.format(len(chars), len(target_chars)))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
target_indices_char = dict((i, c) for i, c in enumerate(target_chars))

print(indices_char)

# --------------------------------------------------------------------
# Segunda passagem: incluir timestamps absolutos (times3)
# --------------------------------------------------------------------

lines = []
caseids = []
timeseqs = []   # tempos entre eventos
timeseqs2 = []  # tempos desde o início (não usados diretamente na avaliação)
timeseqs3 = []  # timestamps absolutos (datetime)

for trace in log:
    case_id = trace.attributes.get("concept:name", None)
    if case_id is None and len(trace) > 0:
        case_id = trace[0].get("CaseID", None)
    caseids.append(case_id)

    line = ''
    times = []
    times2 = []
    times3 = []

    casestarttime = None
    lasteventtime = None

    for event in trace:
        ts = event["CompleteTimestamp"]
        act_id = int(event["ActivityID"])

        if casestarttime is None:
            casestarttime = ts
            lasteventtime = ts

        line += chr(act_id + ascii_offset)

        timesincelastevent = ts - lasteventtime
        timesincecasestart = ts - casestarttime

        midnight = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = ts - midnight

        timediff = int(timesincelastevent.total_seconds())
        timediff2 = int(timesincecasestart.total_seconds())
        # timediff = log(timediff+1)  # mantido como comentário, como no original

        times.append(timediff)
        times2.append(timediff2)
        times3.append(ts)

        lasteventtime = ts

    lines.append(line)
    timeseqs.append(times)
    timeseqs2.append(times2)
    timeseqs3.append(times3)

numlines = len(lines)
print(f"Número de casos (segunda passagem): {numlines}")

# Folds novamente (com times3 agora)
elems_per_fold = int(round(numlines / 3))

fold1 = lines[:elems_per_fold]
fold1_c = caseids[:elems_per_fold]
fold1_t = timeseqs[:elems_per_fold]
fold1_t2 = timeseqs2[:elems_per_fold]
fold1_t3 = timeseqs3[:elems_per_fold]

fold2 = lines[elems_per_fold:2 * elems_per_fold]
fold2_c = caseids[elems_per_fold:2 * elems_per_fold]
fold2_t = timeseqs[elems_per_fold:2 * elems_per_fold]
fold2_t2 = timeseqs2[elems_per_fold:2 * elems_per_fold]
fold2_t3 = timeseqs3[elems_per_fold:2 * elems_per_fold]

fold3 = lines[2 * elems_per_fold:]
fold3_c = caseids[2 * elems_per_fold:]
fold3_t = timeseqs[2 * elems_per_fold:]
fold3_t2 = timeseqs2[2 * elems_per_fold:]
fold3_t3 = timeseqs3[2 * elems_per_fold:]

lines = fold3
caseids = fold3_c
lines_t = fold3_t
lines_t2 = fold3_t2  # não usados diretamente
lines_t3 = fold3_t3

# --------------------------------------------------------------------
# Parâmetros de predição
# --------------------------------------------------------------------

predict_size = 1  # próxima atividade

# --------------------------------------------------------------------
# Carregar modelo
# --------------------------------------------------------------------

model = load_model(MODEL_PATH)

# --------------------------------------------------------------------
# Funções auxiliares
# --------------------------------------------------------------------

def encode(sentence, times, times3, maxlen=maxlen):
    """
    Codifica um prefixo (sentence) + tempos em um tensor X[1, maxlen, num_features]
    """
    num_features = len(chars) + 5
    X = np.zeros((1, maxlen, num_features), dtype=np.float32)
    leftpad = maxlen - len(sentence)
    times2 = np.cumsum(times)
    for t_idx, char in enumerate(sentence):
        midnight = times3[t_idx].replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = times3[t_idx] - midnight
        multiset_abstraction = Counter(sentence[:t_idx + 1])
        for c in chars:
            if c == char:
                X[0, t_idx + leftpad, char_indices[c]] = 1
        X[0, t_idx + leftpad, len(chars)] = t_idx + 1
        X[0, t_idx + leftpad, len(chars) + 1] = times[t_idx] / divisor
        X[0, t_idx + leftpad, len(chars) + 2] = times2[t_idx] / divisor2
        X[0, t_idx + leftpad, len(chars) + 3] = timesincemidnight.seconds / 86400
        X[0, t_idx + leftpad, len(chars) + 4] = times3[t_idx].weekday() / 7
    return X

def getSymbol(predictions):
    """
    Converte vetor de probabilidades (softmax) no símbolo mais provável.
    """
    maxPrediction = 0
    symbol = ''
    i = 0
    for prediction in predictions:
        if prediction >= maxPrediction:
            maxPrediction = prediction
            symbol = target_indices_char[i]
        i += 1
    return symbol

one_ahead_gt = []
one_ahead_pred = []

two_ahead_gt = []
two_ahead_pred = []

three_ahead_gt = []
three_ahead_pred = []

# --------------------------------------------------------------------
# Predição e avaliação
# --------------------------------------------------------------------

results_path = os.path.join("output_files", "results", f"next_activity_and_time_{eventlog}")
os.makedirs(os.path.dirname(results_path), exist_ok=True)

with open(results_path, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow([
        "CaseID", "Prefix length", "Groud truth", "Predicted",
        "Levenshtein", "Damerau", "Jaccard",
        "Ground truth times", "Predicted times", "RMSE", "MAE"
    ])

    for prefix_size in range(2, maxlen):
        print(prefix_size)
        for line, caseid, times, times3 in zip(lines, caseids, lines_t, lines_t3):
            times.append(0)  # placeholder como no original

            cropped_line = ''.join(line[:prefix_size])
            cropped_times = times[:prefix_size]
            cropped_times3 = times3[:prefix_size]

            if '!' in cropped_line:
                # caso já terminaria antes desse prefixo
                continue

            ground_truth = ''.join(line[prefix_size:prefix_size + predict_size])
            ground_truth_t = times[prefix_size:prefix_size + predict_size]

            predicted = ''
            predicted_t = []

            for i in range(predict_size):
                if len(ground_truth) <= i:
                    continue

                enc = encode(cropped_line, cropped_times, cropped_times3)
                y = model.predict(enc, verbose=0)
                y_char = y[0][0]
                y_t = y[1][0][0]

                prediction = getSymbol(y_char)
                cropped_line += prediction

                if y_t < 0:
                    y_t = 0
                cropped_times.append(y_t)

                # reescala tempo previsto
                y_t = y_t * divisor
                cropped_times3.append(cropped_times3[-1] + timedelta(seconds=y_t))
                predicted_t.append(y_t)

                # salvar estatísticas one-/two-/three-step-ahead
                if i == 0:
                    if len(ground_truth_t) > 0:
                        one_ahead_pred.append(y_t)
                        one_ahead_gt.append(ground_truth_t[0])
                if i == 1:
                    if len(ground_truth_t) > 1:
                        two_ahead_pred.append(y_t)
                        two_ahead_gt.append(ground_truth_t[1])
                if i == 2:
                    if len(ground_truth_t) > 2:
                        three_ahead_pred.append(y_t)
                        three_ahead_gt.append(ground_truth_t[2])

                if prediction == '!':
                    print('! predicted, end case')
                    break

                predicted += prediction

            output = []
            if len(ground_truth) > 0:
                output.append(caseid)
                output.append(prefix_size)
                output.append(ground_truth)
                output.append(predicted)
                output.append(1 - distance.nlevenshtein(predicted, ground_truth))
                if len(predicted) > 0 and len(ground_truth) > 0:
                    dls = 1 - (
                        damerau_levenshtein_distance(predicted, ground_truth)
                        / max(len(predicted), len(ground_truth))
                    )
                else:
                    dls = 0
                if dls < 0:
                    dls = 0
                output.append(dls)
                output.append(1 - distance.jaccard(predicted, ground_truth))
                output.append('; '.join(str(x) for x in ground_truth_t))
                output.append('; '.join(str(x) for x in predicted_t))

                # alinhar tamanhos das listas de tempos
                if len(predicted_t) > len(ground_truth_t):
                    predicted_t = predicted_t[:len(ground_truth_t)]
                if len(ground_truth_t) > len(predicted_t):
                    predicted_t.extend(range(len(ground_truth_t) - len(predicted_t)))

                if len(ground_truth_t) > 0 and len(predicted_t) > 0:
                    output.append('')  # RMSE em branco, como no original
                    output.append(metrics.mean_absolute_error(
                        [ground_truth_t[0]], [predicted_t[0]]
                    ))
                    # output.append(metrics.median_absolute_error([ground_truth_t[0]], [predicted_t[0]]))
                else:
                    output.append('')
                    output.append('')
                    output.append('')

                spamwriter.writerow(output)

print(f"Resultados salvos em: {results_path}")