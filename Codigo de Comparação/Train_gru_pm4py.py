"""
Treino de modelo GRU para:
 - Prever a próxima atividade
 - Prever o tempo até a próxima atividade

Leitura do log usando pm4py, com mesma lógica geral do código LSTM original
de Niek Tax / Ilya Verenich, adaptado para Python 3 e GRU.

Autor da adaptação: Lyan Eduardo Sakuno Rodrigues
"""

from __future__ import print_function, division

import os
from collections import Counter

import numpy as np
from pm4py.objects.log.importer.csv import factory as csv_importer

from keras.models import Model
from keras.layers import Input, Dense, GRU
from keras.layers import BatchNormalization
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# ----------------------------------------------------------------------
# Configurações principais
# ----------------------------------------------------------------------

eventlog = "helpdesk.csv"
log_path = os.path.join("data", eventlog)
ascii_offset = 161

OUTPUT_MODELS_DIR = os.path.join("output_files", "models")
OUTPUT_FOLDS_DIR = os.path.join("output_files", "folds")
os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_FOLDS_DIR, exist_ok=True)

print(f"Lendo log com pm4py: {log_path}")

parameters = {
    "case_id": "CaseID",
    "activity_key": "ActivityID",
    "timestamp_key": "CompleteTimestamp"
}
log = csv_importer.apply(log_path, parameters=parameters)

numlines = len(log)
print(f"Número de casos (traces): {numlines}")

# ----------------------------------------------------------------------
# 1ª passagem: construir sequências básicas (lines, timeseqs, timeseqs2)
#    - lines        = sequência de atividades (como string de chars)
#    - timeseqs     = tempo entre eventos consecutivos (segundos)
#    - timeseqs2    = tempo desde o início do caso (segundos)
# ----------------------------------------------------------------------

lines = []
timeseqs = []
timeseqs2 = []

for trace in log:
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

        # codifica atividade como caractere
        line += chr(act_id + ascii_offset)

        # tempos relativos
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

# ----------------------------------------------------------------------
# Divisores (normalização de tempo)
# ----------------------------------------------------------------------

divisor = np.mean([item for sublist in timeseqs for item in sublist])   # tempo médio entre eventos
print('divisor: {}'.format(divisor))

divisor2 = np.mean([item for sublist in timeseqs2 for item in sublist]) # tempo médio desde início do caso
print('divisor2: {}'.format(divisor2))

# ----------------------------------------------------------------------
# Separar em 3 folds (como no código original)
# ----------------------------------------------------------------------

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

# Treino = fold1 + fold2
lines_train = fold1 + fold2
lines_train_t = fold1_t + fold2_t
lines_train_t2 = fold1_t2 + fold2_t2

# ----------------------------------------------------------------------
# 2ª passagem: construir tempos extras (hora do dia, dia da semana)
#    para os mesmos traces, para o treino
# ----------------------------------------------------------------------

lines = []
timeseqs = []
timeseqs2 = []
timeseqs3 = []  # segundos desde meia-noite
timeseqs4 = []  # dia da semana (0-6)

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

numlines2 = len(lines)
assert numlines2 == numlines, "Número de casos mudou entre as passagens!"

# ----------------------------------------------------------------------
# Separar novamente em folds com as novas sequências
# ----------------------------------------------------------------------

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

# Treino = fold1 + fold2
lines = fold1 + fold2
lines_t = fold1_t + fold2_t
lines_t2 = fold1_t2 + fold2_t2
lines_t3 = fold1_t3 + fold2_t3
lines_t4 = fold1_t4 + fold2_t4

# ----------------------------------------------------------------------
# Construção de sentenças (prefixos) e targets (atividade + tempo)
# ----------------------------------------------------------------------

step = 1
sentences = []
sentences_t = []
sentences_t2 = []
sentences_t3 = []
sentences_t4 = []

next_chars = []
next_chars_t = []
next_chars_t2 = []
next_chars_t3 = []
next_chars_t4 = []

softness = 0.0  # label smoothing (0 = one-hot puro)

# adiciona delimitador '!' ao fim de cada caso
lines = [x + '!' for x in lines]

maxlen = max(len(x) for x in lines)

# conjunto de caracteres
chars = [set(x) for x in lines]
chars = list(set().union(*chars))
chars.sort()
target_chars = list(chars)  # inclui '!' também

# para codificação de entrada (não queremos prever '!' como estado normal)
if '!' in chars:
    chars.remove('!')

print(f"total chars: {len(chars)}, target chars: {len(target_chars)}")

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
target_indices_char = dict((i, c) for i, c in enumerate(target_chars))

print("Mapa de caracteres de entrada:", indices_char)

# gerar exemplos de treinamento
for line, line_t, line_t2, line_t3, line_t4 in zip(
    lines, lines_t, lines_t2, lines_t3, lines_t4
):
    for i in range(1, len(line)):  # prefixos começando de tamanho 1
        # prefixo
        sentences.append(line[0:i])
        sentences_t.append(line_t[0:i])
        sentences_t2.append(line_t2[0:i])
        sentences_t3.append(line_t3[0:i])
        sentences_t4.append(line_t4[0:i])

        # próximo símbolo e seus tempos
        next_chars.append(line[i])

        if i == len(line) - 1:
            # caso especial: fim de sequência, tempo = 0
            next_chars_t.append(0)
            next_chars_t2.append(0)
            next_chars_t3.append(0)
            next_chars_t4.append(0)
        else:
            next_chars_t.append(line_t[i])
            next_chars_t2.append(line_t2[i])
            next_chars_t3.append(line_t3[i])
            next_chars_t4.append(line_t4[i])

print('nb sequences:', len(sentences))

# ----------------------------------------------------------------------
# Vetorização em tensores (X, y_a, y_t)
# ----------------------------------------------------------------------

print('Vectorization...')
num_features = len(chars) + 5
print('num features: {}'.format(num_features))

X = np.zeros((len(sentences), maxlen, num_features), dtype=np.float32)
y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
y_t = np.zeros((len(sentences)), dtype=np.float32)

for i, sentence in enumerate(sentences):
    leftpad = maxlen - len(sentence)
    next_t = next_chars_t[i]
    sentence_t = sentences_t[i]
    sentence_t2 = sentences_t2[i]
    sentence_t3 = sentences_t3[i]
    sentence_t4 = sentences_t4[i]

    for t, char in enumerate(sentence):
        multiset_abstraction = Counter(sentence[:t + 1])  # mantido como no original

        # codificação one-hot da atividade atual
        for c in chars:
            if c == char:
                X[i, t + leftpad, char_indices[c]] = 1

        # atributos adicionais
        X[i, t + leftpad, len(chars)] = t + 1                          # posição no prefixo
        X[i, t + leftpad, len(chars) + 1] = sentence_t[t] / divisor    # tempo desde último evento
        X[i, t + leftpad, len(chars) + 2] = sentence_t2[t] / divisor2  # tempo desde início do caso
        X[i, t + leftpad, len(chars) + 3] = sentence_t3[t] / 86400.0   # segundos desde meia-noite
        X[i, t + leftpad, len(chars) + 4] = sentence_t4[t] / 7.0       # dia da semana normalizado

    # target de atividade (softmax com smoothing opcional)
    for c in target_chars:
        if c == next_chars[i]:
            y_a[i, target_char_indices[c]] = 1.0 - softness
        else:
            y_a[i, target_char_indices[c]] = softness / (len(target_chars) - 1)

    # target de tempo (normalizado)
    y_t[i] = next_t / divisor

# ----------------------------------------------------------------------
# Construção do modelo GRU multi-saída
# ----------------------------------------------------------------------

print('Build GRU model...')

main_input = Input(shape=(maxlen, num_features), name='main_input')

# camada compartilhada
l1 = GRU(
    100,
    return_sequences=True,
    dropout=0.2,
    name="gru_shared"
)(main_input)
b1 = BatchNormalization(name="bn_shared")(l1)

# ramo para atividade
l2_1 = GRU(
    100,
    return_sequences=False,
    dropout=0.2,
    name="gru_act"
)(b1)
b2_1 = BatchNormalization(name="bn_act")(l2_1)

# ramo para tempo
l2_2 = GRU(
    100,
    return_sequences=False,
    dropout=0.2,
    name="gru_time"
)(b1)
b2_2 = BatchNormalization(name="bn_time")(l2_2)

act_output = Dense(
    len(target_chars),
    activation='softmax',
    name='act_output'
)(b2_1)

time_output = Dense(
    1,
    name='time_output'
)(b2_2)

model = Model(inputs=[main_input], outputs=[act_output, time_output])

opt = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)

model.compile(
    loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae'},
    optimizer=opt,
    loss_weights={'act_output': 1.0, 'time_output': 1.0}
)

model.summary()

# ----------------------------------------------------------------------
# Callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
# ----------------------------------------------------------------------

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=42,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    filepath=os.path.join(OUTPUT_MODELS_DIR, 'gru_model_{epoch:02d}-{val_loss:.2f}.h5'),
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    verbose=1
)

lr_reducer = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    verbose=1,
    mode='auto',
    min_delta=1e-4,
    cooldown=0,
    min_lr=1e-6
)

# ----------------------------------------------------------------------
# Treino
# ----------------------------------------------------------------------

print("Starting training GRU model...")

history = model.fit(
    X,
    {'act_output': y_a, 'time_output': y_t},
    validation_split=0.2,
    verbose=2,
    batch_size=maxlen,
    epochs=500,
    callbacks=[early_stopping, model_checkpoint, lr_reducer]
)

# Salvar modelo final
final_model_path = os.path.join(OUTPUT_MODELS_DIR, 'gru_model_final.h5')
model.save(final_model_path)
print(f"Modelo GRU salvo em: {final_model_path}")