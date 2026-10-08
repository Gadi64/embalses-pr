import io, re, json, time, requests, joblib
import numpy as np, pandas as pd
from math import radians, sin, cos, asin, sqrt
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

DATA, MODELOS = Path("data"), Path("modelos")
DATA.mkdir(exist_ok=True); MODELOS.mkdir(exist_ok=True)
SITIOS = json.loads(Path("sitios.json").read_text(encoding="utf-8"))

HORIZONTE, LAGS = 48, [1, 3, 6, 12, 24]
PRIORIDAD_NIVEL = ["00065", "72379"]
USGS_IV = "https://waterservices.usgs.gov/nwis/iv/"

def ahora():
    # Hora local de PR sin zona, igual que los datos de USGS y Open-Meteo
    return pd.Timestamp.now(tz="America/Puerto_Rico").tz_localize(None)

def _get(url, params, intentos=4):
    for i in range(intentos):
        try:
            r = requests.get(url, params=params, timeout=90)
            if r.status_code < 500 and r.status_code != 429:
                r.raise_for_status()
                return r
        except (requests.ConnectionError, requests.Timeout):
            pass
        if i < intentos - 1:
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"Sin respuesta de {url.split('/')[2]} tras {intentos} intentos")

# ---------- Selección de área ----------
def _km(lat1, lon1, lat2, lon2):
    a = sin(radians(lat2-lat1)/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(radians(lon2-lon1)/2)**2
    return 6371 * 2 * asin(sqrt(a))

def sitios_cercanos(lat, lon, n=3):
    ds = sorted(((_km(lat, lon, s["lat"], s["lon"]), k) for k, s in SITIOS.items()))
    return [{"id": k, "nombre": SITIOS[k]["nombre"], "km": round(d, 1)} for d, k in ds[:n]]

def sitios_en_caja(norte, sur, este, oeste):
    return [k for k, s in SITIOS.items() if sur <= s["lat"] <= norte and oeste <= s["lon"] <= este]

def descubrir_sitios_pr(tipo="LK"):
    """Lista oficial de estaciones activas en PR (LK = lagos/embalses, ST = ríos)."""
    t = _get("https://waterservices.usgs.gov/nwis/site/", {
        "format": "rdb", "stateCd": "PR", "siteType": tipo,
        "hasDataTypeCd": "iv", "siteStatus": "active"}).text
    df = pd.read_csv(io.StringIO(t), comment="#", sep="\t", dtype=str).iloc[1:]
    return {f"s{r.site_no}": {"nombre": r.station_nm.title(), "usgs": r.site_no,
                              "lat": float(r.dec_lat_va), "lon": float(r.dec_long_va)}
            for r in df.itertuples()}

# ---------- Descarga ----------
def parsear_usgs(texto):
    vacio = pd.DataFrame(columns=["datetime", "gage_height", "discharge"])
    try:
        df = pd.read_csv(io.StringIO(texto), comment="#", sep="\t", dtype=str).iloc[1:]
    except Exception:
        return vacio
    if "datetime" not in df.columns:
        return vacio
    cods = {c: c.split("_")[-1] for c in df.columns if re.search(r"_\d{5}$", c)}  # excluye *_cd
    nivel = next((c for p in PRIORIDAD_NIVEL for c, k in cods.items() if k == p), None)
    if nivel is None:
        return vacio
    caudal = next((c for c, k in cods.items() if k == "00060"), None)
    return pd.DataFrame({
        "datetime": pd.to_datetime(df["datetime"], errors="coerce"),
        "gage_height": pd.to_numeric(df[nivel], errors="coerce"),
        "discharge": pd.to_numeric(df[caudal], errors="coerce") if caudal else np.nan,
    }).dropna(subset=["datetime", "gage_height"])

def descargar_usgs(site, desde=None, hasta=None):
    p = {"sites": site, "format": "rdb", "parameterCd": ",".join(PRIORIDAD_NIVEL + ["00060"])}
    if desde is not None:
        p["startDT"] = str(pd.Timestamp(desde).date())
        if hasta is not None:
            p["endDT"] = str(pd.Timestamp(hasta).date())
    else:
        p["period"] = "PT72H"
    try:
        return parsear_usgs(_get(USGS_IV, p).text)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return parsear_usgs("")
        raise

def descargar_meteo(lat, lon, desde, hasta):
    base = {"latitude": lat, "longitude": lon, "timezone": "America/Puerto_Rico",
            "hourly": "temperature_2m,relative_humidity_2m,precipitation"}
    corte = ahora().normalize() - pd.Timedelta(days=7)
    partes = []
    if desde < corte:  # histórico
        partes.append(_get("https://archive-api.open-meteo.com/v1/archive",
            {**base, "start_date": str(desde.date()), "end_date": str(min(hasta, corte).date())}).json()["hourly"])
    partes.append(_get("https://api.open-meteo.com/v1/forecast",   # últimos 7 días + hoy
        {**base, "past_days": 7, "forecast_days": 1}).json()["hourly"])
    df = pd.concat([pd.DataFrame({
        "datetime": pd.to_datetime(h["time"]), "temperature": h["temperature_2m"],
        "humidity": h["relative_humidity_2m"], "precipitation": h["precipitation"]}) for h in partes])
    return df.drop_duplicates("datetime", keep="last")

# ---------- Almacenamiento incremental ----------
def _ruta(clave, tipo): return DATA / f"{SITIOS[clave]['usgs']}_{tipo}.csv"
def _leer(r):
    if not r.exists():
        return pd.DataFrame({"datetime": pd.to_datetime([])})
    d = pd.read_csv(r)
    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce")
    return d
def _unir(a, b):
    return (pd.concat([a, b]).drop_duplicates("datetime", keep="last")
            .sort_values("datetime").reset_index(drop=True))

def actualizar_sitio(clave, desde=None):
    """Descarga solo lo que falta y lo agrega a los CSV del sitio."""
    s, hoy = SITIOS[clave], ahora()
    u, m = _leer(_ruta(clave, "usgs")), _leer(_ruta(clave, "meteo"))
    if desde:
        ini = pd.Timestamp(desde)
    elif len(u):
        ini = u["datetime"].max() - pd.Timedelta(days=1)
    else:
        ini = hoy - pd.Timedelta(days=120)
    ini = ini.normalize()
    for a in pd.date_range(ini, hoy, freq="15D"):
        u = _unir(u, descargar_usgs(s["usgs"], a, min(a + pd.Timedelta(days=15), hoy)))
        time.sleep(1)
    m = _unir(m, descargar_meteo(s["lat"], s["lon"], ini, hoy))
    u.to_csv(_ruta(clave, "usgs"), index=False)
    m.to_csv(_ruta(clave, "meteo"), index=False)
    return len(u)

# ---------- Variables ----------
def construir(u, m, con_objetivo):
    u = u.set_index("datetime")[["gage_height", "discharge"]].resample("1h").mean()
    m = m.set_index("datetime")[["precipitation", "temperature", "humidity"]].resample("1h").agg(
        {"precipitation": lambda x: x.sum(min_count=1), "temperature": "mean", "humidity": "mean"})
    d = u.join(m, how="inner")
    hay_q = d["discharge"].notna().any()
    if not hay_q:
        d = d.drop(columns="discharge")
    for h in LAGS:
        d[f"nivel_{h}h"] = d["gage_height"].shift(h)
        if hay_q: d[f"caudal_{h}h"] = d["discharge"].shift(h)
        d[f"lluvia_acum_{h}h"] = d["precipitation"].rolling(h).sum()
    feats = list(d.columns)
    if con_objetivo:
        d["objetivo"] = d["gage_height"].shift(-HORIZONTE) - d["gage_height"]
    return d, feats

# ---------- Modelo ----------
def entrenar(clave):
    d, feats = construir(_leer(_ruta(clave, "usgs")), _leer(_ruta(clave, "meteo")), True)
    d = d.dropna()
    if len(d) < 500:
        raise ValueError(f"Solo {len(d)} horas utilizables; se necesitan al menos 500 (~3 semanas).")
    corte = int(len(d) * 0.8)
    tr, te = d.iloc[:corte], d.iloc[corte:]
    nuevo = lambda: XGBRegressor(n_estimators=200, learning_rate=0.02, max_depth=2,
                                 subsample=0.7, colsample_bytree=0.8, random_state=42)
    peso = lambda x: np.where(x["gage_height"] > tr["gage_height"].quantile(0.85), 8, 1)
    prueba = nuevo().fit(tr[feats], tr["objetivo"], sample_weight=peso(tr))
    mae = float(mean_absolute_error(te["objetivo"], prueba.predict(te[feats])))
    final = nuevo().fit(d[feats], d["objetivo"], sample_weight=peso(d))   # reentrena con todo
    joblib.dump({"modelo": final, "features": feats, "mae": mae, "fecha": str(ahora())},
                MODELOS / f"{SITIOS[clave]['usgs']}.pkl")
    return {"horas": len(d), "mae_pies": round(mae, 3)}

def predecir(clave):
    u = _leer(_ruta(clave, "usgs"))
    if u.empty or u["datetime"].max() < ahora() - pd.Timedelta(minutes=30):
        actualizar_sitio(clave)
        u = _leer(_ruta(clave, "usgs"))
    art = joblib.load(MODELOS / f"{SITIOS[clave]['usgs']}.pkl")   # FileNotFoundError si no hay modelo
    m = _leer(_ruta(clave, "meteo"))
    lim = ahora() - pd.Timedelta(days=7)
    d, _ = construir(u[u["datetime"] > lim], m[m["datetime"] > lim], False)
    d = d.dropna(subset=art["features"])
    if d.empty or d.index[-1] < ahora() - pd.Timedelta(hours=3):
        raise ValueError("Sin datos recientes suficientes (el sensor puede estar fuera de servicio).")
    fila = d.iloc[[-1]]
    delta = float(art["modelo"].predict(fila[art["features"]])[0])
    nivel = float(fila["gage_height"].iloc[0])
    return {"sitio": SITIOS[clave]["nombre"], "hora_medicion": str(fila.index[0]),
            "hora_objetivo": str(fila.index[0] + pd.Timedelta(hours=HORIZONTE)),
            "nivel_actual": round(nivel, 2), "nivel_predicho_6h": round(nivel + delta, 2),
            "error_tipico": round(art["mae"], 2)}

def historial(clave, horas=48):
    u = _leer(_ruta(clave, "usgs"))
    h = (u[u["datetime"] > ahora() - pd.Timedelta(hours=horas)]
         .set_index("datetime")["gage_height"].resample("1h").mean().dropna())
    return [{"t": str(t), "nivel": round(v, 2)} for t, v in h.items()]
