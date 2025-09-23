import pandas as pd
import numpy as np

mu_values = [0.0, -1.25]
# mu_values = np.round(np.linspace(-3, 4, 21), 2)
file_format = "mu_{mu}.txt"

delta_imp_columns = {}

for mu in mu_values:
    file_name = file_format.format(mu=mu)
    try:
        data = pd.read_csv(file_name, sep=r'\s+')
        delta_imp_columns[f"μ_{mu}"] = data['ΔAvg']
    except FileNotFoundError:
        print(f"File {file_name} not found. Skipping...")
    except KeyError:
        print(f"ΔAvg column not found in {file_name}. Skipping...")

delta_imp_df = pd.DataFrame(delta_imp_columns)

output_file = "delta_grid.csv"
delta_imp_df.to_csv(output_file, index=False)
print(f"Grid saved as '{output_file}'")
