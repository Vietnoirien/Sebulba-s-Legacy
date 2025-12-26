
import urllib.request
import urllib.error
import json
import time

URL_START = "http://localhost:8000/api/start"
URL_STOP = "http://localhost:8000/api/stop"

payload = {
    "model": "scratch",
    "curriculum_mode": "manual",
    "curriculum_stage": 0,
    "config": {
        "rewards": {
            "weights": {
                "velocity": 99.9
            }
        }
    }
}

def post(url, data):
    req = urllib.request.Request(url)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    jsondata = json.dumps(data).encode('utf-8') if data else None
    
    try:
        response = urllib.request.urlopen(req, jsondata)
        return response.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        return f"Error: {e.code} {e.read().decode('utf-8')}"
    except Exception as e:
        return f"Error: {e}"

print(f"Sending POST to {URL_START}")
print(f"Payload: {json.dumps(payload)}")
res = post(URL_START, payload)
print(f"Response: {res}")

time.sleep(2)

print(f"Sending POST to {URL_STOP}")
res = post(URL_STOP, None)
print("Stopped.")
