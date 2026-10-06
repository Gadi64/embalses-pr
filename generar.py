import json, sys, joblib, pandas as pd, core
from pathlib import Path

OUT = Path("docs/data"); (OUT / "historial").mkdir(parents=True, exist_ok=True)
desde = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None

def modelo_viejo(pkl):
    if not pkl.exists(): return True
    f = joblib.load(pkl).get("fecha")
    return f is None or core.ahora() - pd.Timestamp(f) > pd.Timedelta(days=7)

resumen = []
for k, s in core.SITIOS.items():
    item = {"id": k, "nombre": s["nombre"], "lat": s["lat"], "lon": s["lon"], "estado": "ok"}
    try:
        core.actualizar_sitio(k, desde)
        if modelo_viejo(core.MODELOS / f"{s['usgs']}.pkl"):
            core.entrenar(k)
        item.update(core.predecir(k))
        (OUT / "historial" / f"{k}.json").write_text(json.dumps(core.historial(k, 72)))
    except Exception as e:
        item.update(estado="sin_datos", detalle=str(e)[:150])
    resumen.append(item)

(OUT / "predicciones.json").write_text(
    json.dumps({"actualizado": str(core.ahora()), "sitios": resumen}, ensure_ascii=False))
print(sum(i["estado"] == "ok" for i in resumen), "de", len(resumen), "sitios con predicción")
