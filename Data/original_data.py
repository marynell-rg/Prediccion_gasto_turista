import kagglehub
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("bushraqurban/tourism-and-economic-impact")
file_path = path + '/world_tourism_economy_data.csv'
data = pd.read_csv(file_path)