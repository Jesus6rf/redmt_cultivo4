import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Decisión Agrícola: VME + Monte Carlo", layout="wide")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # Asegurar tipos
    if "Fecha_Siembra" in df.columns:
        df["Fecha_Siembra"] = pd.to_datetime(df["Fecha_Siembra"], errors="coerce")
    return df

# ---------- CARGA DE DATOS ----------
DEFAULT_PATH = "datos_rendimiento_cultivos.csv"
data_path = st.sidebar.text_input("Ruta del CSV", DEFAULT_PATH)
if not os.path.exists(data_path):
    st.sidebar.warning("No encuentro el CSV en el repositorio. Colócalo en la raíz o indique la ruta.")
df = load_data(data_path) if os.path.exists(data_path) else pd.DataFrame()

st.title("Análisis de Decisión para Optimizar Rendimiento Agrícola")
st.markdown(
    "Esta app ayuda a comparar estrategias (p. ej., **aplicar fertilización extra** vs **mantener plan actual**) "
    "bajo incertidumbre de rendimiento (alto vs bajo), usando **Matriz de Pagos**, **VME** y **Simulación de Monte Carlo**."
)

# =========================
# 1) PARÁMETROS DEL ESCENARIO
# =========================
st.header("1. Parámetros del Escenario Agrícola")

colA, colB, colC = st.columns([1.2, 1, 1])
with colA:
    st.subheader("Filtrado (opcional) para estimar prevalencia")
    if not df.empty:
        cultivo_sel = st.selectbox("Cultivo", ["(Todos)"] + sorted(df["Cultivo"].unique().tolist()))
        mes = st.selectbox("Mes de siembra", ["(Todos)"] + list(range(1, 13)))
        temp_min, temp_max = st.slider("Rango de temperatura promedio (°C)", -5.0, 40.0, (float(df["Temperatura_Promedio_C"].min()) if not df.empty else 10.0,
                                                                                          float(df["Temperatura_Promedio_C"].max()) if not df.empty else 30.0))
        df_filt = df.copy()
        if cultivo_sel != "(Todos)":
            df_filt = df_filt[df_filt["Cultivo"] == cultivo_sel]
        if mes != "(Todos)" and "Fecha_Siembra" in df_filt.columns:
            df_filt = df_filt[df_filt["Fecha_Siembra"].dt.month == mes]
        if not df_filt.empty:
            df_filt = df_filt[(df_filt["Temperatura_Promedio_C"] >= temp_min) & (df_filt["Temperatura_Promedio_C"] <= temp_max)]
    else:
        cultivo_sel = "(Todos)"
        df_filt = pd.DataFrame()

with colB:
    st.subheader("Definición de ‘alto’ rendimiento")
    # Umbral por percentil (p.ej., top 30%)
    pct = st.slider("Percentil para 'alto' rendimiento", 50, 95, 75, step=1)
    def compute_prev(df_use):
        if df_use.empty:
            return 0.3  # valor por defecto si no hay datos
        thr = np.percentile(df_use["Rendimiento_tn_ha"], pct)
        return float((df_use["Rendimiento_tn_ha"] >= thr).mean()), thr
    if not df_filt.empty:
        p_data, thr_data = compute_prev(df_filt)
    elif not df.empty:
        p_data, thr_data = compute_prev(df)
    else:
        p_data, thr_data = 0.3, None

with colC:
    st.subheader("Ajuste manual")
    p_manual = st.slider("Probabilidad de **ALTO rendimiento** (%)", 0, 100, int(round(p_data*100)))
    p_high = p_manual / 100.0
    p_low = 1 - p_high
    st.write(f"Prob. ALTO: **{p_high:.2f}**, Prob. BAJO: **{p_low:.2f}**")
    if thr_data is not None:
        st.caption(f"Umbral actual para 'alto' rendimiento ≈ {thr_data:.2f} tn/ha (con percentil {pct})")

st.divider()

# =========================
# 2) MATRIZ DE PAGOS
# =========================
st.header("2. Matriz de Pagos (Costos/Beneficios)")
st.markdown(
    "Define los **pagos** (pueden ser costos positivos o beneficios como valores negativos). "
    "Ejemplo: si aplicar fertilización extra cuesta 120 USD/ha, puedes usar **120**. "
    "Si esperas un beneficio neto de 200 USD/ha cuando hay alto rendimiento, usa **-200**."
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Decisión: **Aplicar fertilización extra**")
    C_TC = st.number_input("Pago si el rendimiento es **ALTO** (beneficio negativo, costo positivo)", value=-200.0, step=10.0, format="%.2f")
    C_FP = st.number_input("Pago si el rendimiento es **BAJO** (beneficio/costo)", value=120.0, step=10.0, format="%.2f")
with col2:
    st.subheader("Decisión: **Mantener plan actual**")
    C_FN = st.number_input("Pago si el rendimiento es **ALTO** (oportunidad perdida/costo)", value=0.0, step=10.0, format="%.2f")
    C_TN = st.number_input("Pago si el rendimiento es **BAJO**", value=0.0, step=10.0, format="%.2f")

matriz = pd.DataFrame({
    "Decisión": ["Aplicar fertilización extra", "Mantener plan actual"],
    "ALTO rendimiento": [C_TC, C_FN],
    "BAJO rendimiento": [C_FP, C_TN]
})
st.dataframe(matriz, use_container_width=True)

st.divider()

# =========================
# 3) VALOR MONETARIO ESPERADO (VME)
# =========================
st.header("3. Valor Monetario Esperado (VME)")
VME_trat = p_high * C_TC + p_low * C_FP
VME_obs  = p_high * C_FN + p_low * C_TN
vme_tbl = pd.DataFrame({
    "Decisión": ["Aplicar fertilización extra", "Mantener plan actual"],
    "VME": [VME_trat, VME_obs]
})
st.dataframe(vme_tbl.style.format({"VME": "{:.2f}"}), use_container_width=True)

best_decision = "Aplicar fertilización extra" if VME_trat < VME_obs else "Mantener plan actual"
best_vme = min(VME_trat, VME_obs)
st.success(f"**La mejor estrategia (menor VME)** es: **{best_decision}** con costo/beneficio esperado de **{best_vme:.2f}** por ha.")

st.caption("Recordatorio: valores negativos significan **beneficio** neto; positivos, **costo** neto.")

st.divider()

# =========================
# 4) SIMULACIÓN DE MONTE CARLO
# =========================
st.header("4. Simulación de Monte Carlo (riesgo)")
st.markdown("Se simula la variabilidad de probabilidades y pagos para estimar la **distribución** del resultado de la mejor estrategia.")

colL, colR = st.columns(2)
with colL:
    n_sim = st.number_input("Número de simulaciones", min_value=1000, max_value=100000, value=10000, step=1000)
    st.subheader("Variación de la probabilidad")
    p_sd = st.slider("Desviación (±) de la prob. de ALTO (en puntos porcentuales)", 0, 20, 5)
with colR:
    st.subheader("Variación de pagos (Normal, % del valor)")
    var_pct = st.slider("Desviación estándar relativa (%) aplicada a cada pago", 0, 50, 10)

def sample_normal(base, rel_sd_pct):
    sd = abs(base) * (rel_sd_pct / 100.0)
    # si base = 0, asigna una sd pequeña para que exista variación leve
    if sd == 0:
        sd = 1.0
    return np.random.normal(loc=base, scale=sd)

if st.button("Ejecutar Simulación"):
    rng = np.random.default_rng()
    p_samples = np.clip(rng.normal(p_high, p_sd/100.0, size=n_sim), 0.0, 1.0)

    # pagos simulados por iteración
    C_TC_s = np.array([sample_normal(C_TC, var_pct) for _ in range(n_sim)])
    C_FP_s = np.array([sample_normal(C_FP, var_pct) for _ in range(n_sim)])
    C_FN_s = np.array([sample_normal(C_FN, var_pct) for _ in range(n_sim)])
    C_TN_s = np.array([sample_normal(C_TN, var_pct) for _ in range(n_sim)])

    VME_trat_s = p_samples * C_TC_s + (1 - p_samples) * C_FP_s
    VME_obs_s  = p_samples * C_FN_s + (1 - p_samples) * C_TN_s

    if best_decision == "Aplicar fertilización extra":
        results = VME_trat_s
    else:
        results = VME_obs_s

    mean_cost = results.mean()
    std_cost  = results.std(ddof=1)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Costo/beneficio promedio simulado", f"{mean_cost:,.2f}")
    with c2:
        st.metric("Desviación estándar (riesgo)", f"{std_cost:,.2f}")

    # Histograma
    fig1, ax1 = plt.subplots()
    ax1.hist(results, bins=40)
    ax1.set_title("Distribución simulada del resultado (mejor estrategia)")
    ax1.set_xlabel("Pago por ha (negativo = beneficio)")
    ax1.set_ylabel("Frecuencia")
    st.pyplot(fig1)

    # Curva acumulada (CDF)
    sorted_r = np.sort(results)
    cdf = np.arange(1, len(sorted_r)+1) / len(sorted_r)
    fig2, ax2 = plt.subplots()
    ax2.plot(sorted_r, cdf)
    ax2.set_title("Probabilidad acumulada (CDF)")
    ax2.set_xlabel("Pago por ha")
    ax2.set_ylabel("Probabilidad acumulada")
    st.pyplot(fig2)

    # Prob. de exceder umbral
    st.subheader("Riesgo de exceder umbral")
    umbral = st.number_input("Umbral de costo (pérdida) que te preocupa", value=150.0, step=10.0, format="%.2f")
    prob_exceso = float((results > umbral).mean())
    st.info(f"Probabilidad de superar el umbral {umbral:.2f}: **{prob_exceso:.2%}**")
else:
    st.caption("Configura los parámetros y haz clic en **Ejecutar Simulación**.")
