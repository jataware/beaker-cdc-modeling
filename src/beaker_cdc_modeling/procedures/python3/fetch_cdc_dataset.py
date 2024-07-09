import pandas as pd

# Function to query a specific dataset and retrieve all rows
def query_dataset(endpoint, params):
    base_url = "https://data.cdc.gov/resource/"
    url = f"{base_url}{endpoint}.json"
    all_data = []
    offset = 0
    limit = 1000  # Adjust limit as needed, maximum is typically 1000

    while True:
        # Update parameters with the current offset and limit
        params.update({
            "$limit": limit,
            "$offset": offset
        })

        # Make the GET request to the CDC API
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if not data:  # If no more data is returned, break the loop
                break
            all_data.extend(data)  # Append the retrieved data to the list
            offset += limit  # Increment the offset for the next batch
        else:
            print(f"Failed to retrieve data. HTTP Status code: {response.status_code}")
            break

    return all_data

endpoint = "{{ endpoint }}"
params = {}
data = query_dataset(endpoint, params)

try:
    data = pd.DataFrame(data)
    data
except Exception as e:
    print(f"Failed to convert data to DataFrame: {e}")
    data