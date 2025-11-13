"""
Treino de modelos de Árvore de Decisão para:
 - Prever a próxima atividade
 - Prever o tempo até a próxima atividade

Baseado no mesmo log utilizado pelos modelos LSTM
(helpdesk.csv, formato: CaseID,ActivityID,CompleteTimestamp).

Usa pm4py para leitura do log e scikit-learn para os modelos.

Autor da adaptação: Lyan Eduardo Sakuno Rodrigues
"""

import os

import numpy as np
import pandas as pd
from pm4py.objects.log.importer.csv import factory as csv_importer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from joblib import dump

# ----------------------------------------------------------------------
# Configurações principais
# ----------------------------------------------------------------------

EVENTLOG = "helpdesk.csv"
LOG_PATH = os.path.join("data", EVENTLOG)

OUTPUT_MODELS_DIR = os.path.join("output_files", "models")
OUTPUT_RESULTS_DIR = os.path.join("output_files", "results")
os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42

# ----------------------------------------------------------------------
# 1. Leitura do log com pm4py
# ----------------------------------------------------------------------

print(f"Lendo log: {LOG_PATH}")

parameters = {
    "case_id": "CaseID",
    "activity_key": "ActivityID",
    "timestamp_key": "CompleteTimestamp"
}
log = csv_importer.apply(LOG_PATH, parameters=parameters)

num_cases = len(log)
print(f"Número de casos (traces): {num_cases}")

# ----------------------------------------------------------------------
# 2. Construção do dataset de prefixos
#    Cada linha = um prefixo de caso
#    Target 1: próxima atividade (classe)
#    Target 2: tempo até a próxima atividade (regressão, em segundos)
# ----------------------------------------------------------------------

rows = []

for case_idx, trace in enumerate(log):
    # tenta pegar ID do caso
    case_id = trace.attributes.get("concept:name", None)
    if case_id is None and len(trace) > 0:
        case_id = trace[0].get("CaseID", None)

    if len(trace) < 2:
        # não há próxima atividade pra prever
        continue

    # garante ordenação por timestamp
    sorted_events = sorted(
        trace,
        key=lambda e: e["CompleteTimestamp"]
    )

    # timestamps e atividades
    timestamps = [ev["CompleteTimestamp"] for ev in sorted_events]
    activities = [int(ev["ActivityID"]) for ev in sorted_events]

    # tempos entre eventos consecutivos
    inter_times = []
    for i in range(1, len(sorted_events)):
        dt = (timestamps[i] - timestamps[i - 1]).total_seconds()
        inter_times.append(dt)

    # para cada prefixo (até o penúltimo evento)
    for i in range(len(sorted_events) - 1):
        prefix_acts = activities[:i + 1]
        prefix_ts = timestamps[:i + 1]

        last_ts = prefix_ts[-1]
        first_ts = prefix_ts[0]

        prefix_len = len(prefix_acts)
        last_act = prefix_acts[-1]
        num_distinct_acts = len(set(prefix_acts))

        elapsed_time = (last_ts - first_ts).total_seconds()

        if prefix_len > 1:
            inter_prefix = inter_times[:prefix_len - 1]
            mean_inter_time = float(np.mean(inter_prefix))
            std_inter_time = float(np.std(inter_prefix))
        else:
            mean_inter_time = 0.0
            std_inter_time = 0.0

        last_hour = last_ts.hour
        last_weekday = last_ts.weekday()

        # target: próxima atividade
        next_act = activities[i + 1]
        # target: tempo até próxima atividade
        time_to_next = (timestamps[i + 1] - timestamps[i]).total_seconds()

        row = {
            "case_index": case_idx,
            "case_id": case_id,
            "prefix_len": prefix_len,
            "last_act": last_act,
            "num_distinct_acts": num_distinct_acts,
            "elapsed_time": elapsed_time,
            "mean_inter_time": mean_inter_time,
            "std_inter_time": std_inter_time,
            "last_hour": last_hour,
            "last_weekday": last_weekday,
            "y_next_act": next_act,
            "y_time_to_next": time_to_next,
        }
        rows.append(row)

df = pd.DataFrame(rows)
print(f"Número de exemplos (prefixos): {len(df)}")

if df.empty:
    raise RuntimeError("Nenhum prefixo gerado. Verifique o log de entrada.")

# ----------------------------------------------------------------------
# 3. Split treino/teste por casos (consistente com folds 1+2 vs fold 3)
# ----------------------------------------------------------------------

elems_per_fold = int(round(num_cases / 3))
train_case_limit = 2 * elems_per_fold  # casos com índice < 2*fold = treino

df["set"] = np.where(df["case_index"] < train_case_limit, "train", "test")

df_train = df[df["set"] == "train"].copy()
df_test = df[df["set"] == "test"].copy()

print(f"Exemplos de treino: {len(df_train)}")
print(f"Exemplos de teste: {len(df_test)}")

# ----------------------------------------------------------------------
# 4. Definição de features e targets
# ----------------------------------------------------------------------

feature_cols = [
    "prefix_len",
    "last_act",
    "num_distinct_acts",
    "elapsed_time",
    "mean_inter_time",
    "std_inter_time",
    "last_hour",
    "last_weekday",
]

X_train = df_train[feature_cols].values
X_test = df_test[feature_cols].values

y_train_act = df_train["y_next_act"].values
y_test_act = df_test["y_next_act"].values

y_train_time = df_train["y_time_to_next"].values
y_test_time = df_test["y_time_to_next"].values

# ----------------------------------------------------------------------
# 5. Normalização para o modelo de tempo (opcional, mas útil para generalizar)
# ----------------------------------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------------------
# 6. Treino dos modelos de Árvore de Decisão
#    - Classificador: próxima atividade
#    - Regressor: tempo até próxima atividade
# ----------------------------------------------------------------------

print("Treinando DecisionTreeClassifier para próxima atividade...")
clf_act = DecisionTreeClassifier(
    max_depth=None,        # pode limitar ex: 10 se quiser controlar overfitting
    min_samples_split=2,
    random_state=RANDOM_STATE
)
clf_act.fit(X_train, y_train_act)

print("Treinando DecisionTreeRegressor para tempo até próxima atividade...")
reg_time = DecisionTreeRegressor(
    max_depth=None,
    min_samples_split=2,
    random_state=RANDOM_STATE
)
reg_time.fit(X_train_scaled, y_train_time)

# ----------------------------------------------------------------------
# 7. Avaliação nos dados de teste
# ----------------------------------------------------------------------

# Próxima atividade
y_pred_act = clf_act.predict(X_test)
acc_act = accuracy_score(y_test_act, y_pred_act)

# Tempo até próxima atividade
y_pred_time = reg_time.predict(X_test_scaled)
mae_time = mean_absolute_error(y_test_time, y_pred_time)
rmse_time = mean_squared_error(y_test_time, y_pred_time, squared=False)

print("\n=== Resultados na base de teste (Árvore de Decisão) ===")
print(f"Acurácia (próxima atividade): {acc_act:.4f}")
print(f"MAE (tempo até próxima atividade, segundos): {mae_time:.4f}")
print(f"RMSE (tempo até próxima atividade, segundos): {rmse_time:.4f}")

# ----------------------------------------------------------------------
# 8. Salvar modelos e scaler
# ----------------------------------------------------------------------

model_act_path = os.path.join(OUTPUT_MODELS_DIR, "decision_tree_next_activity.pkl")
model_time_path = os.path.join(OUTPUT_MODELS_DIR, "decision_tree_time_to_next.pkl")
scaler_path = os.path.join(OUTPUT_MODELS_DIR, "decision_tree_time_scaler.pkl")

dump(clf_act, model_act_path)
dump(reg_time, model_time_path)
dump(scaler, scaler_path)

print(f"\nModelos salvos em:")
print(f" - {model_act_path}")
print(f" - {model_time_path}")
print(f"Scaler salvo em: {scaler_path}")

# ----------------------------------------------------------------------
# 9. Salvar resultados detalhados em CSV (útil para análise comparativa)
# ----------------------------------------------------------------------

results_csv_path = os.path.join(
    OUTPUT_RESULTS_DIR,
    f"decision_tree_next_activity_and_time_{EVENTLOG}"
)
df_results = df_test.copy()
df_results["y_pred_next_act"] = y_pred_act
df_results["y_pred_time_to_next"] = y_pred_time

df_results.to_csv(results_csv_path, index=False)
print(f"\nResultados detalhados salvos em: {results_csv_path}")