'''
Train LSTM model on event log using pm4py for log handling.

Author: baseado em Niek Tax / Ilya Verenich
Adaptado para Python 3 + pm4py por Lyan Eduardo Sakuno Rodrigues
'''

from __future__ import print_function, division

from keras.models import Model
from keras.layers import Input, Dense, LSTM
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import BatchNormalization

from collections import Counter
import numpy as np
import copy
import os

from pm4py.objects.log.importer.csv import factory as csv_importer

eventlog = "helpdesk.csv"
log_path = os.path.join("data", eventlog)

ascii_offset = 161

########################################################################################
# Leitura do log usando pm4py e construção das sequências
########################################################################################

print(f"Lendo log com pm4py: {log_path}")

parameters = {
    "case_id": "CaseID",
    "activity_key": "ActivityID",
    "timestamp_key": "CompleteTimestamp"
}
log = csv_importer.apply(log_path, parameters=parameters)

lines = []      # sequências de atividades codificadas como chars
timeseqs = []   # tempos entre eventos consecutivos
timeseqs2 = []  # tempo desde o início do caso

numlines = 0

for trace in log:
    line = ''
    times = []
    times2 = []

    casestarttime = None
    lasteventtime = None

    for i, event in enumerate(trace):
        # pm4py já converte o timestamp para datetime
        ts = event["CompleteTimestamp"]
        act_id = int(event["ActivityID"])

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

        lasteventtime = ts

    lines.append(line)
    timeseqs.append(times)
    timeseqs2.append(times2)
    numlines += 1

print(f"Número de casos (traces): {numlines}")

########################################
# Estatísticas de tempo (divisores)
########################################

divisor = np.mean([item for sublist in timeseqs for item in sublist])   # tempo médio entre eventos
print('divisor: {}'.format(divisor))
divisor2 = np.mean([item for sublist in timeseqs2 for item in sublist]) # tempo médio desde o início
print('divisor2: {}'.format(divisor2))

#########################################################################################################
# Separação em 3 folds
#########################################################################################################

elems_per_fold = int(round(numlines / 3))

fold1 = lines[:elems_per_fold]
fold1_t = timeseqs[:elems_per_fold]
fold1_t2 = timeseqs2[:elems_per_fold]

fold2 = lines[elems_per_fold:2 * elems_per_fold]
fold2_t = timeseqs[elems_per_fold:2 * elems_per_fold]
fold2_t2 = timeseqs2[elems_per_fold:2 * elems_per_fold]

fold3 = lines[2 * elems_per_fold:]
fold3_t = timeseqs[2 * elems_per_fold:]
fold3_t2 = timeseqs2[2 * elems_per_fold:]

# treino em fold1 + fold2
lines = fold1 + fold2
lines_t = fold1_t + fold2_t
lines_t2 = fold1_t2 + fold2_t2

########################################################################################
# Segunda passagem: tempos adicionais (hora do dia, dia da semana) usando pm4py
########################################################################################

lines = []
timeseqs = []
timeseqs2 = []
timeseqs3 = []  # tempo desde meia-noite (segundos)
timeseqs4 = []  # dia da semana (0-6)

numlines = 0

for trace in log:
    line = ''
    times = []
    times2 = []
    times3 = []
    times4 = []

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
        timediff3 = timesincemidnight.seconds
        timediff4 = ts.weekday()

        times.append(timediff)
        times2.append(timediff2)
        times3.append(timediff3)
        times4.append(timediff4)

        lasteventtime = ts

    lines.append(line)
    timeseqs.append(times)
    timeseqs2.append(times2)
    timeseqs3.append(times3)
    timeseqs4.append(times4)
    numlines += 1

print(f"Número de casos (segunda passagem): {numlines}")

elems_per_fold = int(round(numlines / 3))
fold1 = lines[:elems_per_fold]
fold1_t = timeseqs[:elems_per_fold]
fold1_t2 = timeseqs2[:elems_per_fold]
fold1_t3 = timeseqs3[:elems_per_fold]
fold1_t4 = timeseqs4[:elems_per_fold]

fold2 = lines[elems_per_fold:2 * elems_per_fold]
fold2_t = timeseqs[elems_per_fold:2 * elems_per_fold]
fold2_t2 = timeseqs2[elems_per_fold:2 * elems_per_fold]
fold2_t3 = timeseqs3[elems_per_fold:2 * elems_per_fold]
fold2_t4 = timeseqs4[elems_per_fold:2 * elems_per_fold]

fold3 = lines[2 * elems_per_fold:]
fold3_t = timeseqs[2 * elems_per_fold:]
fold3_t2 = timeseqs2[2 * elems_per_fold:]
fold3_t3 = timeseqs3[2 * elems_per_fold:]
fold3_t4 = timeseqs4[2 * elems_per_fold:]

# aqui você pode manter a lógica original de salvar folds em CSV, gerar sentenças,
# vetorização (X, y_a, y_t) e definição/treino do modelo exatamente como já estava
# na versão em Python 3 sem pm4py.