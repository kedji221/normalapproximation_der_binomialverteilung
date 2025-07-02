import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Binomial Dichtefunktion
def binomial_pmf(n, k, p):
    return math.comb(n, k) * (p**k) * ((1 - p) ** (n - k))

# Binomial kumulative Werte als Kurve
def binomial_cdf_curve(n, p):
    x = np.arange(n + 1)
    pmf = np.array([binomial_pmf(n, k, p) for k in x])
    cdf = np.cumsum(pmf)
    return x, cdf

# Binomial  komplementäre kumulative Verteilung Werte als Kurve
def binomial_sf_curve(n, p):
    x = np.arange(n + 1)
    pmf = np.array([binomial_pmf(n, k, p) for k in x])
    sf = 1 - np.cumsum(pmf) + pmf  # P(X >= k)
    return x, sf

# Normal CDF Werte als Kurve
def normal_cdf_curve(n, p):
    mu = n * p
    sigma = math.sqrt(n * p * (1 - p))
    x = np.linspace(0, n, 500)
    cdf = norm.cdf(x + 0.5, mu, sigma)  # Stetigkeitskorrektur
    return x, cdf

# Normal  komplementäre kumulative Verteilung Werte als Kurve
def normal_sf_curve(n, p):
    mu = n * p
    sigma = math.sqrt(n * p * (1 - p))
    x = np.linspace(0, n, 500)
    sf = 1 - norm.cdf(x - 0.5, mu, sigma)  # Stetigkeitskorrektur für Survival
    return x, sf

st.title("🔍 Normalapproximation der Binomialverteilung (mit Stetigkeitskorrektur)")

# --- Sidebar ---
with st.sidebar:
    st.header("Einstellungen")
    n = st.slider("Anzahl der Versuche (n)", 1, 300, 30)
    p = st.slider("Erfolgswahrscheinlichkeit (p)", 0.01, 1.0, 0.5, step=0.01)
    k = st.slider("Anzahl der Erfolge (k)", 0, n, int(n // 2))
    use_continuity = st.checkbox("Stetigkeitskorrektur anwenden", value=True)

# Erwartungswert & Standardabweichung
mu = n * p
sigma = math.sqrt(n * p * (1 - p))

st.latex(fr"\text{{Erwartungswert: }} \mu = {n} \cdot {p:.2f} = {mu:.2f}")
st.latex(fr"\text{{Standardabweichung: }} \sigma = \sqrt{{{n} \cdot {p:.2f} \cdot (1 - {p:.2f})}} = {sigma:.2f}")

if sigma < 3:
    st.error("σ muss größer als 3 sein (σ > 3)")
else:
    # Wahrscheinlichkeiten
    bin_exact = binomial_pmf(n, k, p)
    bin_cum = sum(binomial_pmf(n, i, p) for i in range(0, k+1))
    bin_sf = sum(binomial_pmf(n, i, p) for i in range(k, n+1))
    norm_eq = norm.cdf(k + 0.5, mu, sigma) - norm.cdf(k - 0.5, mu, sigma) if use_continuity else 0
    norm_leq_val = norm.cdf(k + 0.5, mu, sigma) if use_continuity else norm.cdf(k, mu, sigma)
    norm_geq_val = 1 - norm.cdf(k - 0.5, mu, sigma) if use_continuity else 1 - norm.cdf(k, mu, sigma)

    st.subheader("📊 Wahrscheinlichkeiten")
    st.markdown(f"**Binomial:** P(X = {k}) = `{bin_exact:.4f}`")
    st.markdown(f"**Binomial kumuliert:** P(X ≤ {k}) = `{bin_cum:.4f}`")
    st.markdown(f"**Binomial Survival:** P(X ≥ {k}) = `{bin_sf:.4f}`")
    st.markdown(f"**Normalapproximation:** P(X = {k}) ≈ `{norm_eq:.4f}`")
    st.markdown(f"P(X ≤ {k}) ≈ `{norm_leq_val:.4f}`")
    st.markdown(f"P(X ≥ {k}) ≈ `{norm_geq_val:.4f}`")

    # Werte für die Plots
    x_vals = np.arange(n + 1)
    bin_vals = [binomial_pmf(n, x, p) for x in x_vals]
    norm_x = np.linspace(max(0, mu - 4 * sigma), min(n, mu + 4 * sigma), 500)
    norm_y = norm.pdf(norm_x, mu, sigma)

    # --- Grafik 1: Einzelwahrscheinlichkeit ---
    st.markdown("### 1. Einzelwahrscheinlichkeit $P(X = k)$")
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(x_vals, bin_vals, color='skyblue', label="Binomialverteilung", alpha=0.6, zorder=1)
    ax1.plot(norm_x, norm_y, color='red', label="Normalverteilung", linewidth=2, zorder=2)
    if use_continuity:
        x_fill = np.linspace(k - 0.5, k + 0.5, 100)
        y_fill = norm.pdf(x_fill, mu, sigma)
        ax1.fill_between(x_fill, y_fill, color='orange', alpha=0.6, label="Stetigkeitskorrektur", zorder=3)
        ax1.vlines([k - 0.5, k + 0.5], 0, max(y_fill), color='green', linestyle='--', alpha=0.8)
    ax1.axvline(mu, color='black', linestyle='--', linewidth=2, label='Mittelwert μ', zorder=4)
    ax1.set_xlabel("Anzahl Erfolge (k)")
    ax1.set_ylabel("Wahrscheinlichkeit / Dichte")
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()
    st.pyplot(fig1)

    # --- Grafik 2: P(X ≤ k) und kumulative Kurven ---
    st.markdown("### 2. Kumulative Wahrscheinlichkeit $P(X \\leq k)$ und Kurvenvergleich")
    col1, col2 = st.columns(2)

    # Links: Kumulative Wahrscheinlichkeit
    with col1:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(x_vals[:k+1], bin_vals[:k+1], color='dodgerblue', label=f"Binomial P(X≤{k})", alpha=0.7, zorder=1)
        ax2.bar(x_vals[k+1:], bin_vals[k+1:], color='lightgrey', alpha=0.3, zorder=0)
        ax2.plot(norm_x, norm_y, color='red', label="Normalverteilung", linewidth=2, zorder=2)
        x_cum = np.linspace(max(0, mu - 4 * sigma), k + 0.5, 300)
        y_cum = norm.pdf(x_cum, mu, sigma)
        ax2.fill_between(x_cum, y_cum, color='orange', alpha=0.5, label="Stetigkeitskorrektur (Fläche)", zorder=3)
        ax2.vlines(k + 0.5, 0, norm.pdf(k + 0.5, mu, sigma), color='green', linestyle='--', alpha=0.8)
        ax2.axvline(mu, color='black', linestyle='--', linewidth=2, label='Mittelwert μ', zorder=4)
        ax2.set_xlabel("Anzahl Erfolge (k)")
        ax2.set_ylabel("Wahrscheinlichkeit / Dichte")
        ax2.legend(fontsize=9)
        ax2.grid(True, linestyle='--', alpha=0.6, zorder=0)
        plt.tight_layout()
        st.pyplot(fig2)

    # Rechts: Kurvenvergleich kumuliert
    with col2:
        xb, yb = binomial_cdf_curve(n, p)
        xn, yn = normal_cdf_curve(n, p)
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.plot(xn, yn, color="red", linewidth=2, label="Normalapproximation (kumuliert)")
        ax3.step(xb, yb, color="blue", linewidth=2, where="post", label="Binomialverteilung (kumuliert)")
        ax3.axvline(mu, color='black', linestyle='--', linewidth=2, label='Mittelwert μ')
        ax3.set_xlim(0, n)
        ax3.set_ylim(0, 1.05)
        ax3.set_xlabel("Anzahl Erfolge (k)")
        ax3.set_ylabel("Kumulative Wahrscheinlichkeit")
        ax3.legend(fontsize=9)
        ax3.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig3)

    # --- Grafik 3: P(X ≥ k) und  komplementäre kumulative Verteilung-Kurven ---
    st.markdown("### 3. - komplementäre kumulative Verteilungsfunktion $P(X \\geq k)$ und -Kurvenvergleich")
    col3, col4 = st.columns(2)

    # Links: Survival-Funktion
    with col3:
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        ax4.bar(x_vals[k:], bin_vals[k:], color='dodgerblue', label=f"Binomial P(X≥{k})", alpha=0.7, zorder=1)
        ax4.bar(x_vals[:k], bin_vals[:k], color='lightgrey', alpha=0.3, zorder=0)
        ax4.plot(norm_x, norm_y, color='red', label="Normalverteilung", linewidth=2, zorder=2)
        x_sf = np.linspace(k - 0.5, min(n, mu + 4 * sigma), 300)
        y_sf = norm.pdf(x_sf, mu, sigma)
        ax4.fill_between(x_sf, y_sf, color='orange', alpha=0.5, label="Stetigkeitskorrektur (Fläche)", zorder=3)
        ax4.vlines(k - 0.5, 0, norm.pdf(k - 0.5, mu, sigma), color='green', linestyle='--', alpha=0.8)
        ax4.axvline(mu, color='black', linestyle='--', linewidth=2, label='Mittelwert μ', zorder=4)
        ax4.set_xlabel("Anzahl Erfolge (k)")
        ax4.set_ylabel("Wahrscheinlichkeit / Dichte")
        ax4.legend(fontsize=9)
        ax4.grid(True, linestyle='--', alpha=0.6, zorder=0)
        plt.tight_layout()
        st.pyplot(fig4)

    # Rechts: Survival-Kurvenvergleich
    with col4:
        xb, yb = binomial_sf_curve(n, p)
        xn, yn = normal_sf_curve(n, p)
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        ax5.plot(xn, yn, color="red", linewidth=2, label="Normalapproximation (Survival)")
        ax5.step(xb, yb, color="blue", linewidth=2, where="post", label="Binomialverteilung (Survival)")
        ax5.axvline(mu, color='black', linestyle='--', linewidth=2, label='Mittelwert μ')
        ax5.set_xlim(0, n)
        ax5.set_ylim(0, 1.05)
        ax5.set_xlabel("Anzahl Erfolge (k)")
        ax5.set_ylabel("Survival-Wahrscheinlichkeit")
        ax5.legend(fontsize=9)
        ax5.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig5)
