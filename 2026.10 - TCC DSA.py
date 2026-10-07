# -*- coding: utf-8 -*-
"""
TCC - Fatores associados ao tempo de análise regulatória de softwares médicos na ANVISA
Script consolidado (versão 7): reproduz todas as tabelas, figuras e números do texto.

Correções em relação à versão anterior (parecer de avaliação):
  - estoque pendente e entradas incluem as petições ainda em andamento;
  - verificação de completude também para o grupo de comparação;
  - regressão quantílica sem variáveis da base de produtos, com IC por bootstrap;
  - classificação "até 30 dias" comparada com a classe majoritária, com AUC;
  - DiD em escala logarítmica, sensibilidades, placebo e estudo de eventos;
  - série temporal interrompida estimada no nível da petição.

Entradas (mesma pasta):
  CICLO_ANALISE_PETICOES_PRODUTO_SAUDE_CORRELATOS.CSV        (finalizadas; datas MM/DD/AAAA)
  CICLO_ANALISE_PETICOES_ANDAMENTO_PRODUTO_SAUDECORRELATOS.CSV (em andamento; datas DD/MM/AAAA)
  TA_PRODUTO_SAUDE_SITE_1.xlsx                                 (produtos registrados)
Saídas: pasta saida_v7 (resultados.json, tabelas.xlsx, figuras .png)
"""
import json, os, warnings, logging
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, mean_absolute_error, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore"); logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

ARQ_FIN = "CICLO_ANALISE_PETICOES_PRODUTO_SAUDE_CORRELATOS.CSV"
ARQ_AND = "CICLO_ANALISE_PETICOES_ANDAMENTO_PRODUTO_SAUDECORRELATOS.CSV"
ARQ_TA = "TA_PRODUTO_SAUDE_SITE_1.xlsx"
PASTA = "saida_v7"; os.makedirs(PASTA, exist_ok=True)
INI, FIM, RDC = pd.Timestamp("2020-01-01"), pd.Timestamp("2026-04-01"), pd.Timestamp("2022-07-01")
NUCLEO = {"80271", "80272", "80273", "80274", "80275", "80276", "80298", "80299", "80310"}
IVD = {"80298", "80299", "80310"}
FILAS = ["Alterações", "Notificações", "Registros", "Revalidações"]
ETAPAS = ["Fila de Análise", "Análise em Andamento", "Exigência", "Análise de Cumprimento de Exigência",
          "Finalização", "Recurso", "Outras Etapas", "Sobrestado Externo", "Sobrestado Anvisa"]
R, TAB = {}, {}
rng = np.random.default_rng(42)
plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 9,
                     "axes.edgecolor": "black", "axes.linewidth": 1.5, "axes.grid": False,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})


def salva(fig, nome):
    fig.tight_layout(); fig.savefig(os.path.join(PASTA, nome), bbox_inches="tight"); plt.close(fig)


def chave(s):
    d = s.astype(str).str.replace(r"\D", "", regex=True)
    return np.where(d.str.len() == 17, d.str[:15], d)


# =========================================================== 1. bases
c = pd.read_csv(ARQ_FIN, sep=";", encoding="cp1252", dtype=str)
c.columns = [x.strip("#") for x in c.columns]
R["linhas_lidas"] = len(c)
c = c[c.DESC_GRUPO_ETAPA_CICLO_ANALISE.isin(ETAPAS)].copy()
R["linhas_malformadas"] = R["linhas_lidas"] - len(c)
for col in ["DATA_PRIMEIRA_FINALIZACAO", "DATA_INI_OCORRENCIA_GRP_ETAPA", "DATA_FIM_OCORRENCIA_GRP_ETAPA"]:
    c[col] = pd.to_datetime(c[col], format="%m/%d/%Y %H:%M:%S", errors="coerce")
c["dur"] = (c.DATA_FIM_OCORRENCIA_GRP_ETAPA - c.DATA_INI_OCORRENCIA_GRP_ETAPA).dt.total_seconds() / 86400
g = c.groupby("NUM_EXPEDIENTE_PETICAO")
p = (c.sort_values("DATA_INI_OCORRENCIA_GRP_ETAPA").groupby("NUM_EXPEDIENTE_PETICAO").first()
     [["NUM_PROCESSO_PETICAO", "COD_ASSUNTO_PETICAO", "DESC_SITUACAO_ATUAL_PETICAO", "DATA_PRIMEIRA_FINALIZACAO",
       "DESC_AREA_INTERESSE", "DESC_FILA_ANALISE", "DESC_SUB_FILA_LISTA_ANALISE"]])
p["entrada"] = g.DATA_INI_OCORRENCIA_GRP_ETAPA.min()
piv = c.pivot_table(index="NUM_EXPEDIENTE_PETICAO", columns="DESC_GRUPO_ETAPA_CICLO_ANALISE", values="dur", aggfunc="sum").fillna(0)
p = p.join(piv.rename(columns={"Fila de Análise": "t_fila", "Análise em Andamento": "t_analise", "Finalização": "t_final",
                               "Exigência": "t_exig"})[["t_fila", "t_analise", "t_final", "t_exig"]])
p["n_exig"] = c[c.DESC_GRUPO_ETAPA_CICLO_ANALISE == "Exigência"].groupby("NUM_EXPEDIENTE_PETICAO").size()
p["n_exig"] = p.n_exig.fillna(0)
# tempo em dias com fração: as datas da base trazem horário
p["t"] = (p.DATA_PRIMEIRA_FINALIZACAO - p.entrada).dt.total_seconds() / 86400
p["ano"] = p.entrada.dt.year
p["periodo"] = np.where(p.entrada >= RDC, "Posterior", "Anterior")
p["recusa"] = p.DESC_SITUACAO_ATUAL_PETICAO.isin(["Publicado indeferimento", "Não anuído"])
p = p.reset_index()

a = pd.read_csv(ARQ_AND, sep=";", encoding="cp1252", dtype=str)
a["d"] = pd.to_datetime(a.DATA_INI_OCORRENCIA_GRP_ETAPA, format="%d/%m/%Y %H:%M:%S")
ae = a.groupby("NUM_EXPEDIENTE_PETICAO").agg(ent=("d", "min"), cod=("COD_ASSUNTO_PETICAO", "first"),
                                              area=("DESC_AREA_INTERESSE", "first"), fila=("DESC_FILA_ANALISE", "first"),
                                              sub=("DESC_SUB_FILA_LISTA_ANALISE", "first"))
R["samd_aberta_mais_antiga"] = str(ae[ae.cod.isin(NUCLEO)].ent.min().date())
eq_and = ae[(ae.area == "Dispositivos Médicos") & ae["sub"].str.contains("Equipamento", na=False) & ae.fila.isin(FILAS)
            & (ae.ent >= INI) & (ae.ent < FIM) & ~ae.cod.isin(NUCLEO)]
R["equip_abertas_na_janela"] = len(eq_and)
dm_and = ae[ae.area == "Dispositivos Médicos"]
R["area_abertas_2025_2026"] = int(((dm_and.ent >= "2025-01-01") & (dm_and.ent < FIM)).sum())

ta = pd.read_excel(ARQ_TA, sheet_name=0, dtype=str)
ta["K"] = chave(ta.NUMERO_PROCESSO)
p["K"] = chave(p.NUM_PROCESSO_PETICAO)
p = p.merge(ta.drop_duplicates("K")[["K", "CLASSE_RISCO", "CNPJ_DETENTOR_REGISTRO_CADASTRO", "NOME_PAIS_FABRIC"]], on="K", how="left")

# ---- estoque e entradas da área, incluindo petições em andamento
pa = p[p.DESC_AREA_INTERESSE == "Dispositivos Médicos"]
E_fin = np.sort(pa.entrada.values); F_fin = np.sort(pa.DATA_PRIMEIRA_FINALIZACAO.dropna().values)
E_and = np.sort(dm_and.ent.values)
E_all = np.sort(np.concatenate([E_fin, E_and]))
estoque = lambda d: (np.searchsorted(E_fin, d) - np.searchsorted(F_fin, d, side="right") + np.searchsorted(E_and, d))
estoque_antigo = lambda d: (np.searchsorted(E_fin, d) - np.searchsorted(F_fin, d, side="right"))
for d in ["2025-07-01", "2026-01-01", "2026-03-31"]:
    x = np.datetime64(d); R[f"subestimacao_{d}"] = float(1 - estoque_antigo(x) / estoque(x))

# =========================================================== 2. núcleo SaMD
n = p[p.COD_ASSUNTO_PETICAO.isin(NUCLEO) & (p.entrada >= INI) & (p.entrada < FIM)].copy()
n["tipo"] = n.DESC_FILA_ANALISE.replace({"Alterações": "Alteração de notificação", "Notificações": "Notificação",
                                         "Registros": "Registro", "Revalidações": "Revalidação"})
pais = n.NOME_PAIS_FABRIC.str.upper().str.strip()
n["origem"] = np.select([pais.isna(), pais == "BRASIL"], ["Não identificada", "Nacional"], "Estrangeira")
n = n.sort_values("entrada").reset_index(drop=True)
n["exp"] = n.groupby("CNPJ_DETENTOR_REGISTRO_CADASTRO").cumcount()
n.loc[n.CNPJ_DETENTOR_REGISTRO_CADASTRO.isna(), "exp"] = np.nan
n["exigencia"] = np.where(n.n_exig > 0, "Sim", "Não")
n["estq"] = estoque(n.entrada.values) / 1000
R["n_nucleo"] = len(n); R["n_com_produto"] = int(n.CLASSE_RISCO.notna().sum())
R["spearman_estoque"] = float(stats.spearmanr(n.estq, n.t).statistic)
R["n_80217_80226"] = int(p[p.COD_ASSUNTO_PETICAO.isin(["80217", "80226"]) & (p.entrada >= INI) & (p.entrada < FIM)].shape[0])

# =========================================================== 3. regressão quantílica com bootstrap
def qr_boot(formula, df, B=500):
    fit = smf.quantreg(formula, df).fit(q=.5, max_iter=5000)
    bs = []
    for _ in range(B):
        s = df.sample(len(df), replace=True, random_state=rng.integers(1e9))
        try: bs.append(smf.quantreg(formula, s).fit(q=.5, max_iter=5000).params)
        except Exception: pass
    bs = pd.DataFrame(bs)
    return pd.DataFrame({"coef": fit.params, "lo": bs.quantile(.025), "hi": bs.quantile(.975),
                         "p_boot": [2 * min((bs[k] <= 0).mean(), (bs[k] >= 0).mean()) for k in fit.params.index]}), int(fit.nobs)


f1 = 't ~ C(tipo, Treatment("Notificação")) + C(periodo, Treatment("Anterior")) + estq'
f2 = f1 + " + C(exigencia)"
q1, n1 = qr_boot(f1, n); q2, n2 = qr_boot(f2, n)
TAB["quantilica_m1"] = q1.reset_index(); TAB["quantilica_m2"] = q2.reset_index(); R["n_qr"] = [n1, n2]
# modelo 3: subconjunto com correspondência, com origem e experiência
m3d = n[n.CLASSE_RISCO.notna()].copy(); m3d["exp_log"] = np.log1p(m3d.exp)
m3d = m3d[m3d.origem != "Não identificada"]
q3, n3 = qr_boot(f1 + ' + C(origem, Treatment("Estrangeira")) + exp_log', m3d, B=300)
TAB["quantilica_m3"] = q3.reset_index(); R["n_qr3"] = n3
# OLS log: efeito do período com e sem estoque
n["log_t"] = np.log(n.t)
o0 = smf.ols('log_t ~ C(tipo, Treatment("Notificação")) + C(periodo, Treatment("Anterior"))', n).fit(cov_type="HC3")
o1 = smf.ols('log_t ~ C(tipo, Treatment("Notificação")) + C(periodo, Treatment("Anterior")) + estq', n).fit(cov_type="HC3")
kp = 'C(periodo, Treatment("Anterior"))[T.Posterior]'
R.update(ols_r2=[o0.rsquared, o1.rsquared], per_sem=float(np.exp(o0.params[kp]) - 1),
         per_com=float(np.exp(o1.params[kp]) - 1), estq_pct=float(np.exp(o1.params["estq"]) - 1))

# =========================================================== 4. Lei de Little (anual, área)
rows = []
for m in pd.period_range("2020-01", "2026-03", freq="M"):
    a0, b0 = np.datetime64(m.to_timestamp()), np.datetime64((m + 1).to_timestamp())
    dias = (b0 - a0) / np.timedelta64(1, "D")
    ent = np.searchsorted(E_all, b0) - np.searchsorted(E_all, a0)
    sai = np.searchsorted(F_fin, b0) - np.searchsorted(F_fin, a0)
    est = np.mean([estoque(np.datetime64(d)) for d in pd.date_range(m.to_timestamp(), (m + 1).to_timestamp(), freq="7D")])
    rows.append({"ano": m.year, "ent": ent, "sai": sai, "est": est, "little": est / (sai / dias)})
L = pd.DataFrame(rows).groupby("ano").agg(ent=("ent", "mean"), sai=("sai", "mean"), est=("est", "mean"), little=("little", "mean"))
L["obs"] = pa[pa.t >= 0].groupby("ano").t.mean().reindex(L.index)
L.loc[L.index >= 2025, "obs"] = np.nan          # coorte da área incompleta a partir de 2025
L["fila_samd"] = n.groupby("ano").t_fila.mean()
L["fila_por_mil"] = L.fila_samd / (n.groupby("ano").estq.mean())
TAB["little"] = L.reset_index()

# =========================================================== 5. série interrompida no nível da petição
n["mes_idx"] = (n.entrada.dt.year - 2020) * 12 + n.entrada.dt.month - 1
n["pos"] = (n.entrada >= RDC).astype(int)
corte = (2022 - 2020) * 12 + 6
n["mes_pos"] = np.where(n.pos == 1, n.mes_idx - corte, 0)
its1 = smf.ols("t ~ mes_idx + pos + mes_pos", n).fit(cov_type="cluster", cov_kwds={"groups": n.mes_idx})
its2 = smf.ols("t ~ mes_idx + pos + mes_pos + estq", n).fit(cov_type="cluster", cov_kwds={"groups": n.mes_idx})
TAB["its"] = pd.DataFrame({"m1": its1.params, "p1": its1.pvalues, "m2": its2.params, "p2": its2.pvalues}).reset_index()

# =========================================================== 6. diferença em diferenças
eq = p[(p.DESC_AREA_INTERESSE == "Dispositivos Médicos") & p.DESC_SUB_FILA_LISTA_ANALISE.str.contains("Equipamento", na=False)
       & p.DESC_FILA_ANALISE.isin(FILAS) & (p.entrada >= INI) & (p.entrada < FIM) & ~p.COD_ASSUNTO_PETICAO.isin(NUCLEO)].copy()
eq["samd"] = 0; ss = n.copy(); ss["samd"] = 1
cols = ["t", "entrada", "samd", "DESC_FILA_ANALISE", "COD_ASSUNTO_PETICAO", "ano"]
d = pd.concat([ss[cols], eq[cols]]); d["pos"] = (d.entrada >= RDC).astype(int)
d["t90"] = d.t.clip(upper=90); d["lt"] = np.log(d.t.clip(lower=0.5))
R["n_equip"] = len(eq)


def did(df, y="t90"):
    mm = df.groupby(["samd", "pos"])[y].mean(); return (mm[1, 1] - mm[1, 0]) - (mm[0, 1] - mm[0, 0])


def did_ic(df, B=500):
    grp = [x for _, x in df.groupby(["samd", "pos"])]
    bs = [did(pd.concat([x.sample(len(x), replace=True, random_state=rng.integers(1e9)) for x in grp])) for _ in range(B)]
    return did(df), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


TAB["did_celulas"] = d.groupby(["samd", "pos"]).agg(n=("t", "size"), mediana=("t", "median"), t90=("t90", "mean")).reset_index()
dd = {}
dd["Principal (média truncada em 90 dias)"] = did_ic(d)
q = smf.quantreg("t ~ samd * pos", d).fit(q=.5); ci = q.conf_int().loc["samd:pos"]
dd["Mediana (regressão quantílica)"] = (q.params["samd:pos"], ci[0], ci[1])
ol = smf.ols("lt ~ samd * pos + C(DESC_FILA_ANALISE)", d).fit(cov_type="HC3"); ci = ol.conf_int().loc["samd:pos"]
dd["Logaritmo do tempo, ajustado pela fila (%)"] = tuple(100 * (np.exp(x) - 1) for x in (ol.params["samd:pos"], ci[0], ci[1]))
dd["Sem os códigos 80217 e 80226"] = did_ic(d[~d.COD_ASSUNTO_PETICAO.isin(["80217", "80226"])])
dd["Sem os códigos de diagnóstico in vitro"] = did_ic(d[~d.COD_ASSUNTO_PETICAO.isin(IVD)])
pre = d[d.entrada < RDC].copy(); pre["pos"] = (pre.entrada >= "2021-07-01").astype(int)
dd["Placebo: corte em jul. 2021, período anterior"] = did_ic(pre)
TAB["did"] = pd.DataFrame([(k,) + tuple(v) for k, v in dd.items()], columns=["analise", "est", "lo", "hi"])

dq = d[d.entrada >= "2020-07-01"].copy(); dq["q"] = dq.entrada.dt.to_period("Q").astype(str)
ev = smf.ols('t90 ~ C(q) + samd + samd:C(q, Treatment("2022Q2"))', dq).fit(cov_type="HC3")
kk = [x for x in ev.params.index if x.startswith("samd:")]
es = pd.DataFrame({"q": [x.split("[T.")[1][:-1] for x in kk], "coef": ev.params[kk].values,
                   "lo": ev.conf_int().loc[kk, 0].values, "hi": ev.conf_int().loc[kk, 1].values})
es = pd.concat([es, pd.DataFrame([{"q": "2022Q2", "coef": 0, "lo": 0, "hi": 0}])]).sort_values("q").reset_index(drop=True)
TAB["estudo_eventos"] = es
R["es_pre_media"] = float(es[es.q < "2022Q2"].coef.mean()); R["es_pos_media"] = float(es[es.q > "2022Q2"].coef.mean())

# =========================================================== 7. agrupamento
num = ["t", "t_fila", "t_analise", "t_final"]
X = pd.concat([pd.DataFrame(StandardScaler().fit_transform(np.log1p(n[num].clip(lower=0))), columns=num, index=n.index),
               pd.get_dummies(n[["tipo", "periodo", "exigencia", "origem"]], dtype=float)], axis=1)
TAB["silhueta"] = pd.DataFrame([{"k": k, "s": silhouette_score(X, KMeans(k, n_init=20, random_state=42).fit(X).labels_)} for k in range(2, 9)])
km = KMeans(3, n_init=50, random_state=42).fit(X); lab = km.labels_
med = pd.Series(n.t.values).groupby(lab).median().sort_values()
TAB["clusters"] = pd.DataFrame([{"grupo": i + 1, "n": int((lab == k).sum()), "mediana": med[k],
                                 "pct_pos": float((n.periodo.values[lab == k] == "Posterior").mean())} for i, k in enumerate(med.index)])
Z = PCA(2, random_state=42).fit(X); ZZ = Z.transform(X); CC = Z.transform(km.cluster_centers_)

# =========================================================== 8. modelos preditivos
pf = pa[pa.DATA_PRIMEIRA_FINALIZACAO.notna() & (pa.t >= 0)].sort_values("DATA_PRIMEIRA_FINALIZACAO")
fin_, tt_ = pf.DATA_PRIMEIRA_FINALIZACAO.values, pf.t.values
def med_rec(x, w=60):
    lo = np.searchsorted(fin_, x - np.timedelta64(w, "D")); hi = np.searchsorted(fin_, x)
    return np.median(tt_[lo:hi]) if hi > lo else np.nan
n["med_recente"] = [med_rec(x) for x in n.entrada.values]
n["entradas_30d"] = [np.searchsorted(E_all, x) - np.searchsorted(E_all, x - np.timedelta64(30, "D")) for x in n.entrada.values]
n["mes"] = n.entrada.dt.month
XP = pd.get_dummies(n[["tipo"]], dtype=float).join(n[["estq", "med_recente", "entradas_30d", "mes"]]); y = n.t.values; yb = (y > 30).astype(int)
mods = {"Referência: mediana do treino": None, "Referência: mediana recente da área": "rec", "Regressão linear (log)": LinearRegression(),
        "Floresta aleatória": RandomForestRegressor(300, min_samples_leaf=10, criterion="absolute_error", random_state=42),
        "Gradient boosting": HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=.05, max_depth=3, random_state=42)}
dobras = [(2023, "2023-01-01", "2024-01-01"), (2024, "2024-01-01", "2025-01-01"), (2025, "2025-01-01", "2026-04-01")]
reg, cls = [], []
for ano, a0, b0 in dobras:
    tr = (n.entrada < a0).values; te = ((n.entrada >= a0) & (n.entrada < b0)).values
    for nome, m in mods.items():
        if m is None: pr = np.full(te.sum(), np.median(y[tr]))
        elif m == "rec": pr = n.loc[te, "med_recente"].values
        elif "log" in nome: m.fit(XP[tr], np.log(y[tr])); pr = np.exp(m.predict(XP[te]))
        else: m.fit(XP[tr], y[tr]); pr = m.predict(XP[te])
        ae_ = np.abs(y[te] - pr)
        reg.append({"ano": ano, "modelo": nome, "n_teste": int(te.sum()), "mae": ae_.mean(), "mdae": np.median(ae_), "p90": np.percentile(ae_, 90)})
        if nome in ("Referência: mediana recente da área", "Gradient boosting"):
            cls.append({"ano": ano, "modelo": nome, "acc": float(np.mean((pr > 30) == yb[te])), "bal": balanced_accuracy_score(yb[te], pr > 30)})
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, max_depth=3, random_state=42).fit(XP[tr], yb[tr])
    cls.append({"ano": ano, "modelo": "Classe majoritária", "acc": max(yb[te].mean(), 1 - yb[te].mean()), "bal": 0.5})
    cls.append({"ano": ano, "modelo": "AUC do classificador", "acc": roc_auc_score(yb[te], clf.predict_proba(XP[te])[:, 1]), "bal": np.nan})
TAB["pred_reg"] = pd.DataFrame(reg); TAB["pred_cls"] = pd.DataFrame(cls)
gb = mods["Gradient boosting"].fit(XP[tr], y[tr])
pi = permutation_importance(gb, XP[te], y[te], scoring="neg_mean_absolute_error", n_repeats=30, random_state=42)
TAB["importancia"] = pd.Series(pi.importances_mean, index=XP.columns).sort_values(ascending=False).rename("imp").reset_index()

# =========================================================== 9. figuras novas
fig, ax1 = plt.subplots(figsize=(6.3, 3.2))
mm = n.groupby(n.entrada.dt.to_period("M")).agg(med=("t", "median"), estq=("estq", "mean")).reset_index()
mm["ts"] = mm.entrada.dt.to_timestamp()
gridm = n.groupby("mes_idx")[["mes_idx", "pos", "mes_pos"]].first()
fit = its1.predict(gridm)
ax1.plot(mm.ts, mm.med, color="black", linewidth=1, label="Mediana mensal do tempo")
ax1.plot(pd.to_datetime([f"{2020 + i // 12}-{i % 12 + 1:02d}-01" for i in gridm.mes_idx]), fit.values, color="black",
         linewidth=2, linestyle="--", label="Ajuste da série interrompida (média)")
ax1.set_ylabel("Tempo (dias)"); ax1.set_xlabel("Mês de entrada")
ax2 = ax1.twinx(); ax2.spines["right"].set_visible(True)
ax2.plot(mm.ts, mm.estq, color="0.6", linewidth=1.2, label="Estoque da área (mil petições)"); ax2.set_ylabel("Estoque pendente (mil)")
ax1.axvline(RDC, color="0.4", linewidth=.8, linestyle=":")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="upper left")
salva(fig, "serie.png")

fig, ax = plt.subplots(figsize=(6.3, 3.0))
xs = np.arange(len(es)); ax.axhline(0, color="black", linewidth=.8)
ax.errorbar(xs, es.coef, yerr=[es.coef - es.lo, es.hi - es.coef], fmt="o", color="black", ecolor="0.5", capsize=2, markersize=4)
iref = int(es.index[es.q == "2022Q2"][0]); ax.axvline(iref + .5, color="0.4", linestyle=":", linewidth=.8)
ax.set_xticks(xs); ax.set_xticklabels([x.replace("Q", "T") for x in es.q], rotation=90, fontsize=7)
ax.set_xlabel("Trimestre de entrada"); ax.set_ylabel("Diferença SaMD − equipamentos (dias)")
salva(fig, "eventos.png")

fig, (a1, b1) = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1.5, 1]})
ordem = list(mods.keys()); cores = ["0.85", "0.6", "0.4", "#5d7fa3", "#1b4f72"]; hatch = ["", "//", "", "", ""]
rr = TAB["pred_reg"]; w = .16
for i, m in enumerate(ordem):
    v = [rr[(rr.modelo == m) & (rr.ano == k)].mae.iloc[0] for k in (2023, 2024, 2025)]
    a1.bar(np.arange(3) + (i - 2) * w, v, w, color=cores[i], edgecolor="black", linewidth=.5, hatch=hatch[i], label=m.replace("Referência: ", "Ref.: "))
a1.set_xticks(range(3)); a1.set_xticklabels(["2023", "2024", "2025 a mar. 2026"]); a1.set_xlabel("Período de teste"); a1.set_ylabel("Erro absoluto médio (dias)")
a1.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2); a1.text(0, 1.02, "(a)", transform=a1.transAxes)
lab_ = {"med_recente": "Mediana recente da área", "estq": "Estoque pendente", "tipo_Registro": "Tipo: registro", "mes": "Mês de entrada",
        "tipo_Notificação": "Tipo: notificação", "tipo_Revalidação": "Tipo: revalidação", "tipo_Alteração de notificação": "Tipo: alteração de notificação",
        "entradas_30d": "Entradas nos 30 dias anteriores"}
im = TAB["importancia"].sort_values("imp")
b1.barh([lab_[k] for k in im["index"]], im.imp, color="#1b4f72", edgecolor="black", linewidth=.5); b1.axvline(0, color="black", linewidth=.8)
b1.set_xlabel("Aumento do erro ao permutar (dias)"); b1.text(0, 1.02, "(b)", transform=b1.transAxes)
salva(fig, "preditivo.png")

# =========================================================== 10. exportação
with pd.ExcelWriter(os.path.join(PASTA, "tabelas.xlsx")) as wx:
    for k, v in TAB.items(): v.to_excel(wx, sheet_name=k[:31], index=False)
for k, v in TAB.items(): v.to_pickle(os.path.join(PASTA, k + ".pkl"))
json.dump(R, open(os.path.join(PASTA, "resultados.json"), "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(R, ensure_ascii=False, indent=1, default=float))

# =========================================================== 11. tabelas descritivas e figuras restantes
# (Tabelas 3, 4, 5 e A1; Figuras 1, 2, 4 e 6 do texto)
n["grupo_risco"] = np.where(n.COD_ASSUNTO_PETICAO.isin(["80271", "80272", "80298"]), "I-II", "III-IV")
n["faixa_exp"] = pd.cut(n.exp, [-1, 0, 5, np.inf], labels=["Primeira petição", "1 a 5 anteriores", "Mais de 5 anteriores"])
def resumo(df, col):
    return pd.DataFrame([{"categoria": str(k), "n": len(v), "mediana": v.t.median(), "q1": v.t.quantile(.25), "q3": v.t.quantile(.75)}
                         for k, v in df.groupby(col, observed=True)])
def teste(df, col):
    gs = [v.t.values for k, v in df.groupby(col, observed=True) if str(k) != "Não identificada" and len(v) >= 5]
    if len(gs) == 2: return "Mann-Whitney", stats.mannwhitneyu(*gs).pvalue
    return "Kruskal-Wallis", stats.kruskal(*gs).pvalue
fat = []
for nome, col in [("Tipo de petição", "tipo"), ("Grupo de risco", "grupo_risco"), ("Classe de risco", "CLASSE_RISCO"), ("Período", "periodo"),
                  ("Origem do fabricante", "origem"), ("Experiência do detentor", "faixa_exp"), ("Ocorrência de exigência", "exigencia")]:
    r_ = resumo(n, col); tt, pv_ = teste(n, col); r_.insert(0, "fator", nome); r_["teste"] = tt; r_["p"] = pv_; fat.append(r_)
TAB["tab3_fatores"] = pd.concat(fat, ignore_index=True)
TAB["tab4_tipo_periodo"] = pd.DataFrame([{"tipo": k, "n_ant": (v.periodo == "Anterior").sum(), "med_ant": v[v.periodo == "Anterior"].t.median(),
    "n_pos": (v.periodo == "Posterior").sum(), "med_pos": v[v.periodo == "Posterior"].t.median(),
    "p": stats.mannwhitneyu(v[v.periodo == "Anterior"].t, v[v.periodo == "Posterior"].t).pvalue if (v.periodo == "Anterior").sum() >= 5 else np.nan}
    for k, v in n.groupby("tipo")])
ex = []
for nome, v in [("Total", n)] + list(n.groupby("tipo")):
    for var, lab_d in [("exigencia", "Exigência"), ("recusa", "Recusa")]:
        x = (v[var] == "Sim") if var == "exigencia" else v[var]
        tab = pd.crosstab(v.periodo, x); pct = x.groupby(v.periodo).mean() * 100
        if tab.shape == (2, 2):
            chi2, pv_, _, esp = stats.chi2_contingency(tab, correction=False); tst = "Qui-quadrado"
            if (esp < 5).any(): pv_ = stats.fisher_exact(tab)[1]; tst = "Fisher"
            ex.append({"estrato": nome, "desfecho": lab_d, "pct_ant": pct.get("Anterior"), "pct_pos": pct.get("Posterior"), "teste": tst, "p": pv_})
TAB["tab5_exig_recusa"] = pd.DataFrame(ex)
sw = p[p.COD_ASSUNTO_PETICAO.isin(NUCLEO | {"80217", "80226"}) & (p.entrada >= INI) & (p.entrada < FIM)].merge(
    ta.drop_duplicates("K")[["K", "NOME_TECNICO"]], on="K", how="left")
sw["nome_sw"] = sw.NOME_TECNICO.str.contains("oftware|SaMD|navega|planej|imagens|PACS", case=False, na=False)
TAB["tabA1_codigos"] = sw.groupby("COD_ASSUNTO_PETICAO").apply(lambda v: pd.Series({"peticoes": len(v),
    "pct_software": v[v.NOME_TECNICO.notna()].nome_sw.mean() * 100})).reset_index()
R["ate30"] = n.groupby("periodo").t.apply(lambda x: float((x <= 30).mean())).to_dict()
R["ate90"] = n.groupby("periodo").t.apply(lambda x: float((x <= 90).mean())).to_dict()
R["etapas_medianas"] = n.groupby("periodo")[["t_fila", "t_analise", "t_final"]].median().to_dict()
R["volume"] = {"samd_2021": int((n.ano == 2021).sum()), "samd_2023_25": float(n.ano.isin([2023, 2024, 2025]).sum() / 3),
               "eq_2021": int((eq.ano == 2021).sum()), "eq_2023_25": float(eq.ano.isin([2023, 2024, 2025]).sum() / 3)}

fig, ax = plt.subplots(figsize=(6.3, 3.2))
ea = n.groupby("ano")[["t_fila", "t_analise", "t_final", "t_exig"]].mean().reset_index(); base_ = np.zeros(len(ea))
for col, lb, cor in [("t_fila", "Fila de análise", "0.2"), ("t_analise", "Análise técnica", "0.5"), ("t_final", "Finalização", "0.75"), ("t_exig", "Exigência", "0.9")]:
    ax.bar(ea.ano.astype(str), ea[col], bottom=base_, color=cor, edgecolor="black", linewidth=.5, label=lb); base_ += ea[col].values
ax.set_xlabel("Ano de entrada"); ax.set_ylabel("Tempo médio (dias)"); ax.legend(frameon=False, fontsize=8); salva(fig, "etapas.png")

nm = n[n.tipo == "Notificação"].groupby(["ano", "DESC_SITUACAO_ATUAL_PETICAO"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(6.3, 3.0))
na_ = nm.get("Não anuído", 0); pi_ = nm.get("Publicado indeferimento", 0)
ax.bar(nm.index.astype(str), na_, color="0.3", edgecolor="black", linewidth=.5, label="Não anuído")
ax.bar(nm.index.astype(str), pi_, bottom=na_, color="0.75", edgecolor="black", linewidth=.5, label="Publicado indeferimento")
ax.set_xlabel("Ano de entrada"); ax.set_ylabel("Notificações recusadas (n)"); ax.legend(frameon=False); salva(fig, "recusa.png")

da = d.groupby(["ano", "samd"]).t.median().unstack()
fig, ax = plt.subplots(figsize=(6.3, 3.2))
ax.plot(da.index, da[1], color="black", marker="o", label="SaMD (núcleo)")
ax.plot(da.index, da[0], color="0.55", marker="s", linestyle="--", label="Equipamentos")
ax.axvline(2022.5, color="0.4", linewidth=.8, linestyle=":"); ax.set_xlabel("Ano de entrada"); ax.set_ylabel("Mediana do tempo (dias)")
ax.legend(frameon=False); salva(fig, "did.png")

nomes = {med.index[0]: "Grupo 1: rápido, anterior à norma", med.index[1]: "Grupo 2: intermediário", med.index[2]: "Grupo 3: lento, fila congestionada"}
fig, (a1, b1) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
for i, k in enumerate(med.index):
    mk, cor = [("o", "#1b4f72"), ("s", "#b9770e"), ("^", "#922b21")][i]
    a1.scatter(ZZ[lab == k, 0], ZZ[lab == k, 1], s=10, marker=mk, color=cor, alpha=.55, linewidths=0, label=f"{nomes[k]} (n = {(lab == k).sum()})")
    a1.scatter(CC[k, 0], CC[k, 1], s=140, marker="X", color=cor, edgecolor="black", linewidth=1.2, zorder=5)
ev_ = Z.explained_variance_ratio_ * 100
a1.set_xlabel(f"Componente principal 1 ({ev_[0]:.0f}% da variância)"); a1.set_ylabel(f"Componente principal 2 ({ev_[1]:.0f}% da variância)")
a1.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.2)); a1.text(0, 1.02, "(a)", transform=a1.transAxes)
sl = TAB["silhueta"]; b1.plot(sl.k, sl.s, color="black", marker="o"); b1.set_ylim(0, .4)
b1.set_xlabel("Número de grupos (k)"); b1.set_ylabel("Coeficiente de silhueta"); b1.text(0, 1.02, "(b)", transform=b1.transAxes)
salva(fig, "kmeans.png")

with pd.ExcelWriter(os.path.join(PASTA, "tabelas.xlsx")) as wx:
    for k, v in TAB.items(): v.to_excel(wx, sheet_name=k[:31], index=False)
json.dump(R, open(os.path.join(PASTA, "resultados.json"), "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ["ate30", "ate90", "etapas_medianas", "volume"]}, ensure_ascii=False, default=float))