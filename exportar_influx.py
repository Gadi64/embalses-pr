
import os
import json
from pathlib import Path
from influxdb_client import InfluxDBClient

cliente = InfluxDBClient(
    url=os.environ["INFLUX_URL"],
    token=os.environ["INFLUX_TOKEN"],
    org=os.environ["INFLUX_ORG"]
)

consulta = f'''
from(bucket: "{os.environ["INFLUX_BUCKET"]}")
  |> range(start: -48h)
  |> filter(fn: (r) =>
      r._measurement == "nivel_embalse" and
      r._field == "distancia_cm"
  )
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
  |> sort(columns: ["_time"])
'''

datos = [
    {
        "hora": registro.get_time().isoformat(),
        "distancia_cm": round(float(registro.get_value()), 2),
        "elevacion_ft": None
    }
    for tabla in cliente.query_api().query(consulta)
    for registro in tabla.records
]

destino = Path("docs/data/sensor/ultrasonico.json")
destino.parent.mkdir(parents=True, exist_ok=True)

with destino.open("w", encoding="utf-8") as archivo:
    json.dump(datos, archivo, ensure_ascii=False, indent=2)

print(f"Se exportaron {len(datos)} mediciones a {destino}")

cliente.close()
