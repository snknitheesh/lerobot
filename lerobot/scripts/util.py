import json 

with open("follower_arm.json", "r") as f:
    config = json.load(f)
    config = config.get("default", {})
    
print(config)