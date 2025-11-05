'''
This script takes as input the LSTM or RNN weights found by Train.py
and evaluates:
 - suffix prediction (continuation of the case)
 - remaining cycle time

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
import matplotlib.pyplot as plt  # mantido como no original (caso queira plotar)
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
# Primeira passagem: construir sequências e tempos
# --------------------------------------------------------------------

lines = []      # sequência de atividades (como string de chars)
caseids = []    # IDs dos casos
timeseqs = []   # tempos entre eventos consecutivos (em segundos)
timeseqs2 = []  # tempos desde o início do caso (em segundos)
timeseqs3 = []  # timestamps absolutos (datetime) de cada evento

for trace in log:
    # tenta pegar o id do caso do atributo do trace
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
        ts = event["CompleteTimestamp"]           # datetime
        act_id = int(event["ActivityID"])         # numérico

        if casestarttime is None:
            casestarttime = ts
            lasteventtime = ts

        # codifica atividade como caractere
        line += chr(act_id + ascii_offset)

        # diferenças de tempo
        timesincelastevent = ts - lasteventtime
        timesincecasestart = ts - casestarttime

        timediff = int(timesincelastevent.total_seconds())
        timediff2 = int(timesincecasestart.total_seconds())

        times.append(timediff)
        times2.append(timediff2)
        times3.append(ts)

        lasteventtime = ts

    lines.append(line)
    timeseqs.append(times)
    timeseqs2.append(times2)
    timeseqs3.append(times3)

numlines = len(lines)
print(f"Número de casos (traces): {numlines}")

# --------------------------------------------------------------------
# Cálculo dos divisores (normalização de tempos)
# --------------------------------------------------------------------

divisor = np.mean([item for sublist in timeseqs for item in sublist])
print('divisor: {}'.format(divisor))

divisor2 = np.mean([item for sublist in timeseqs2 for item in sublist])
print('divisor2: {}'.format(divisor2))

# divisor3: média da diferença entre duração do caso e cada evento (em segundos)
divisor3 = np.mean([
    np.mean([x[-1] - y for y in x]) for x in timeseqs2
])
print('divisor3: {}'.format(divisor3))

# --------------------------------------------------------------------
# Separar em 3 folds (como no original)
# --------------------------------------------------------------------

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

# folds 1 + 2 = base usada para vocabulário e maxlen (como no treino)
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

# adiciona delimitador '!' ao fim de cada sequência
lines_train = [x + '!' for x in lines_train]
maxlen = max(len(x) for x in lines_train)

# conjunto de caracteres
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
# Definir conjunto de teste (fold 3)
# --------------------------------------------------------------------

lines = fold3
caseids = fold3_c
lines_t = fold3_t
lines_t2 = fold3_t2
lines_t3 = fold3_t3

# Parâmetros de predição
predict_size = maxlen

# --------------------------------------------------------------------
# Carregar modelo treinado
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
        multiset_abstraction = Counter(sentence[:t_idx + 1])  # mantido como no original, embora não seja usado diretamente
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

results_path = os.path.join("output_files", "results", f"suffix_and_remaining_time_{eventlog}")
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
        for line, caseid, times, times2, times3 in zip(lines, caseids, lines_t, lines_t2, lines_t3):
            # times: tempo entre eventos sucessivos
            # times2: tempo desde o início do caso (acumulado)
            times.append(0)  # placeholder como no original

            cropped_line = ''.join(line[:prefix_size])
            cropped_times = times[:prefix_size]
            cropped_times3 = times3[:prefix_size]

            if len(times2) < prefix_size:
                # caso já terminou antes desse prefixo
                continue

            ground_truth = ''.join(line[prefix_size:prefix_size + predict_size])
            ground_truth_t_start = times2[prefix_size - 1]
            case_end_time = times2[-1]
            ground_truth_t = case_end_time - ground_truth_t_start  # tempo restante real

            predicted = ''
            total_predicted_time = 0

            for i in range(predict_size):
                enc = encode(cropped_line, cropped_times, cropped_times3)
                y = model.predict(enc, verbose=0)  # [y_char, y_t]
                y_char = y[0][0]
                y_t = y[1][0][0]

                prediction = getSymbol(y_char)
                cropped_line += prediction

                if y_t < 0:
                    y_t = 0
                cropped_times.append(y_t)

                if prediction == '!':
                    # fim de caso previsto, para de predizer
                    one_ahead_pred.append(total_predicted_time)
                    one_ahead_gt.append(ground_truth_t)
                    print('! predicted, end case')
                    break

                # reescala tempo previsto
                y_t = y_t * divisor3
                cropped_times3.append(cropped_times3[-1] + timedelta(seconds=y_t))
                total_predicted_time = total_predicted_time + y_t
                predicted += prediction

            output = []
            if len(ground_truth) > 0:
                output.append(caseid)
                output.append(prefix_size)
                output.append(ground_truth)
                output.append(predicted)
                # similaridade de Levenshtein normalizada
                output.append(1 - distance.nlevenshtein(predicted, ground_truth))
                # Damerau-Levenshtein Similarity
                if len(predicted) > 0 and len(ground_truth) > 0:
                    dls = 1 - (
                        damerau_levenshtein_distance(predicted, ground_truth)
                        / max(len(predicted), len(ground_truth))
                    )
                else:
                    dls = 0
                if dls < 0:
                    dls = 0  # proteção contra valores negativos em algumas plataformas
                output.append(dls)
                # Jaccard
                output.append(1 - distance.jaccard(predicted, ground_truth))
                # tempos
                output.append(ground_truth_t)
                output.append(total_predicted_time)
                output.append('')  # RMSE não está sendo calculado aqui
                output.append(metrics.mean_absolute_error(
                    [ground_truth_t], [total_predicted_time]
                ))
                # output.append(metrics.median_absolute_error([ground_truth_t], [total_predicted_time]))

                spamwriter.writerow(output)

print(f"Resultados salvos em: {results_path}")