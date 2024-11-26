import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Iris.csv")
print (df.head(10))

df.describe()

df.info()

#plotting a histogram
fig, axes = plt.subplots(2, 2, figsize=(10,10))

axes[0,0].set_title("Sepal Length")
axes[0,0].hist(df['SepalLengthCm'], bins=10,edgecolor='black');
axes[0,0].set_xlabel('Sepal Length (cm)') # Added xlabel to axes
axes[0,0].set_ylabel('Frequency') # Added ylabel to axes

axes[0,1].set_title("Sepal Width")
axes[0,1].hist(df['SepalWidthCm'], bins=20,color='green',edgecolor='black');
axes[0,1].set_xlabel('Sepal Width (cm)') # Added xlabel to axes
axes[0,1].set_ylabel('Frequency') # Added ylabel to axes

axes[1,0].set_title("Petal Length")
axes[1,0].hist(df['PetalLengthCm'], bins=15,color='red',edgecolor='black');
axes[1,0].set_xlabel('Petal Length (cm)') # Added xlabel to axes
axes[1,0].set_ylabel('Frequency') # Added ylabel to axes

axes[1,1].set_title("Petal Width")
axes[1,1].hist(df['PetalWidthCm'], bins=18,color='orange',edgecolor='black');
axes[1,1].set_xlabel('Petal Width (cm)') # Added xlabel to axes
axes[1,1].set_ylabel('Frequency') # Added ylabel to axes

plt.tight_layout()
plt.show()
