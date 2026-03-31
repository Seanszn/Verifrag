import os
import requests
from dotenv import load_dotenv

# Load .env
load_dotenv()

API_TOKEN = os.getenv("COURTLISTENER_API_TOKEN")

if not API_TOKEN:
    raise ValueError("API token not found. Check your .env file.")

BASE_URL = "https://www.courtlistener.com/api/rest/v4/search/"

def test_courtlistener():
    headers = {
        "Authorization": f"Token {API_TOKEN}",
        "Accept": "application/json"
    }

    params = {
        "q": "negligence",   # test query
        "page_size": 10      # 🔥 pull 10 documents
    }

    print("🔎 Sending request to CourtListener...\n")

    response = requests.get(BASE_URL, headers=headers, params=params)

    if response.status_code != 200:
        print("❌ Request failed:", response.status_code)
        print(response.text)
        return

    data = response.json()

    results = data.get("results", [])

    print(f"✅ Retrieved {len(results)} documents\n")

    for i, item in enumerate(results, 1):
        case_name = item.get("caseName", "Unknown Case")
        court = item.get("court", "Unknown Court")
        date = item.get("dateFiled", "Unknown Date")
        url = item.get("absolute_url", "")

        print(f"{i}. {case_name}")
        print(f"   Court: {court}")
        print(f"   Date: {date}")
        print(f"   URL: https://www.courtlistener.com{url}")
        print("-" * 60)

if __name__ == "__main__":
    test_courtlistener()

    