import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Rossi dataset
data = pd.read.xlsx('RADCURE_Clinical_v04_20241219')
