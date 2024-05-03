import requests
import json
 
url = "http://localhost:11434/api/generate"
 
payload = json.dumps({
  "model": "llama3",
  "prompt": "Why is the sky blue?",
  "stream": False
})
headers = {
  'Content-Type': 'application/json'
}
 
response = requests.request("POST", url, headers=headers, data=payload)
 
print(response.text)