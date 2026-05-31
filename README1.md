Processing method:
+ Input 4 raw datasets
  - raw_data_concrete_iit.txt: IIT Bhubaneswar lab data (space-separated text file)
  - Concrete_Data.csv: UCI dataset with verbose column names
  - concrete_compressive_strength_dataset.csv: standard concrete dataset  
  - Data_UHPC.csv: Ultra High Performance Concrete data
+ Each dataset has its columns renamed to a consistent standard (e.g. slag → ggbs, ash → flyash, strength → cs)
+ Any columns that don't exist in a particular dataset (e.g. silica_fume, nano_silica) are filled with 0
+ All datasets are forced into the same 15 columns so they can be merged
+ UHPC rows are flagged with is_uhpc = 1, everything else is_uhpc = 0 
Result:
+ Two engineered features are added to every row:
    - binder: sum of all cementitious materials (cement + flyash + ggbs + silica_fume + etc.)
    - water_binder_ratio: water divided by binder, a key concrete mix design parameter
+3 csv files are produced:
    - normal_concrete.csv: all non-UHPC rows
    - uhpc_concrete.csv: UHPC rows only
    - combined_concrete.csv: all data together
