import requests

# Function to list filtered datasets
def list_filtered_datasets(keyword):
    catalog_url = "https://data.cdc.gov/api/views/metadata/v1"
    response = requests.get(catalog_url)
    if response.status_code == 200:
        datasets = response.json()
        filtered_datasets = [dataset for dataset in datasets if keyword.lower() in dataset['name'].lower()]
        for dataset in filtered_datasets:
            print(f"Title: {dataset['name']}")
            print(f"Endpoint: {dataset['id']}")
            print("------")
        return filtered_datasets
    else:
        print(f"Failed to retrieve datasets. HTTP Status code: {response.status_code}")
        return []

# Filter datasets based on query
filtered_datasets = list_filtered_datasets('{{ query }}')
filtered_datasets