import requests

API_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions"

# HEADERS = {"Authorization": f"Bearer {API_KEY}"}

aqi = 1
weather = 25

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json(), response.status_code

response, statusCode = query({
    "messages": [
        {
            "role": "user",
            "content": f"""
                        Given the air quality index (AQI) of {aqi} and the weather conditions described as "{weather}" which is in celcius,, generate a short two-sentence advisory:
                        1. Describe the air quality situation concisely.
                        2. Provide a simple recommendation on outdoor activity.

                        Keep it clear and direct, without unnecessary details.
                        """
        }
    ],
    "max_tokens": 50,
    "model": "mistralai/Mistral-7B-Instruct-v0.3"
})

# print(response["choices"][0]["message"]['content'])
print(statusCode)