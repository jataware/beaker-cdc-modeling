import pandas as pd

# Function to query a specific dataset
def query_dataset(endpoint, params):
    dataset = []
    base_url = "https://data.cdc.gov/resource/"
    url = f"{base_url}{endpoint}.json"
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        for item in data:
            dataset.append(item)
        return dataset
    else:
        print(f"Failed to retrieve data. HTTP Status code: {response.status_code}")

endpoint = '{{ endpoint }}' 
example_params = {
    # TODO: Add parameters to filter the dataset as needed
    # "$limit": 10, 
}
data = query_dataset(endpoint, example_params)

try:
    data = pd.DataFrame(data)
    print(data)
except Exception as e:
    print(f"Failed to convert data to DataFrame: {e}")
    print(data)