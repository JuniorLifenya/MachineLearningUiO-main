import kagglehub

# Download latest version
path = kagglehub.dataset_download("darkmatternet/s-and-p-500-stocks-25-years-of-data-updated-daily")

print("Path to dataset files:", path)