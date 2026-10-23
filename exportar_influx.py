import os, json
from influxdb_client import InfluxDBClient

c = InfluxDBClient(url=os.environ["INFLUX_URL"],
                   token=os.environ["INFLUX_TOKEN"],
                   org=os.environ["INFLUX_ORG"])
q = f'''
from(bucket: "{os.environ["INFLUX_BUCKET"]}")
  |> range(start: -48h)
  |> filter(fn: (r) => r._measurement == "nivel_embalse" and r._field == "nivel")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
'''
datos = [{"t": r.get_time().isoformat(), "nivel": round(r.get_value(), 2)}
         for tabla in c.query_api().query(q) for r in tabla.records]
os.makedirs("data", exist_ok=True)
json.dump(datos, open("data/ultrasonico.json", "w"))
