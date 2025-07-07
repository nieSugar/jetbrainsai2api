import requests
from openai import OpenAI
import json


def get_weather(latitude, longitude):
    
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    
    print(f"Tool>\t You are using get_weather, Args: (latitude: {latitude}, longitude: {longitude}); Result: {data['current']['temperature_2m']}")
    
    return data["current"]["temperature_2m"]


client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-your-custom-key-here")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

messages = [{"role": "user", "content": "What's the weather like in Paris today?"}]

completion = client.chat.completions.create(
    model="anthropic-claude-4-sonnet", messages=messages, tools=tools
)
print(f"User>\t {messages[0]['content']}")

tool_call = completion.choices[0].message.tool_calls[0]
args = json.loads(tool_call.function.arguments)

result = get_weather(args["latitude"], args["longitude"])
# print(result)

messages.append(completion.choices[0].message)  # append model's function call message
messages.append(
    {  # append result message
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result),
    }
)

completion_2 = client.chat.completions.create(
    model="anthropic-claude-4-sonnet",
    messages=messages,
    tools=tools,
)
print(f"Model>\t {completion_2.choices[0].message.content}")
