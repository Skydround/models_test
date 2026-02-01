
import os

try:
    with open('.env', 'r') as f:
        lines = f.readlines()
    
    with open('.env', 'w') as f:
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                # Remove existing quotes if any
                value = value.strip('"\'')
                f.write(f'{key}="{value}"\n')
    print("Fixed .env file format")
except Exception as e:
    print(f"Error: {e}")
