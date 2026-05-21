import requests

API_KEY = "579b464db66ec23bdd000001e290ee4ae7994780538f23665ca841bc"
RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"

BASE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"


def get_mandi_prices(state=None, commodity=None, market=None):
    try:
        params = {
            "api-key": API_KEY,
            "format": "json",
            "limit": 20
        }

        if state:
            params["filters[state]"] = state
        if commodity:
            params["filters[commodity]"] = commodity
        if market:
            params["filters[market]"] = market

        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        return data.get("records", [])

    except Exception as e:
        print("Mandi API Error:", e)
        return []