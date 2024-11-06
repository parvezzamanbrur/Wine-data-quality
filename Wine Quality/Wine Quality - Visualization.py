import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


df= pd.read_csv("Data/WineQT.csv")
df=df.drop(['Id'], axis=1)
df['quality'] = df['quality']-3

def get_column_summary(dataframe, cat_threshold=10, high_card_threshold=20):
    """
    Summarizes the characteristics of columns in a DataFrame.

    Parameters:
    - dataframe: pandas DataFrame
    - cat_threshold: int, threshold for numeric columns to be treated as categorical based on unique values
    - high_card_threshold: int, threshold for categorical columns to be considered high cardinality

    Returns:
    - summary_df: pandas DataFrame summarizing column types and counts
    """

    # Identify categorical columns based on data type
    categorical_cols = dataframe.select_dtypes(include='object').columns.tolist()  # data type is object

    # Identify numeric columns that behave like categories
    numeric_as_categorical = [col for col in dataframe.select_dtypes(exclude='object').columns
                              if dataframe[col].nunique() < cat_threshold]

    # Identify high cardinality categorical columns
    high_cardinality_categorical = [col for col in categorical_cols
                                    if dataframe[col].nunique() > high_card_threshold]

    # Combine actual categorical columns with numeric columns that act like categories
    categorical_cols += numeric_as_categorical

    # Identify purely numerical columns, excluding those treated as categorical
    numerical_cols = [col for col in dataframe.select_dtypes(exclude='object').columns
                      if col not in numeric_as_categorical]

    # Create a DataFrame to summarize column information
    summary_df = pd.DataFrame({
        'Metric': [
            'DataFrame Shape',
            'Total Columns',
            'Categorical Columns',
            'Numerical Columns',
            'High Cardinality Categorical',
            'Numerical but Categorical'
        ],
        'Count': [
            f"{dataframe.shape}",
            dataframe.shape[1],
            len(categorical_cols),
            len(numerical_cols),
            len(high_cardinality_categorical),
            len(numeric_as_categorical)
        ],
        'Columns': [
            '',
            '',
            categorical_cols,
            numerical_cols,
            high_cardinality_categorical,
            numeric_as_categorical
        ]
    })

    return summary_df


# Apply the function to generate the summary DataFrame
column_summary_df = get_column_summary(df)  # Assuming df is your DataFrame variable
print(column_summary_df)

plt.figure(figsize = (12, 12))
sns.heatmap(df.corr(), annot = True, cmap='Spectral')
plt.show()

plt.figure(figsize=(20,10))
corr = df.corr()
sns.heatmap(corr,annot=True,cmap='Blues')
plt.show()


df_category=df.copy()
df_category=df_category.sort_values(by='quality', ascending=True)

df_category["Quality Category"]=df_category["quality"]
df_category.replace({"Quality Category": {0: "Terrible", 1: "Very Poor", 2: "Poor", 3: "Good", 4: "Very Good", 5: "Excellent"}}, inplace=True)

# Create subplots
f, ax = plt.subplots(1, 2, figsize=(22, 8))

# Pie chart
df_category["Quality Category"].value_counts().plot.pie(
    autopct='%1.1f%%',
    ax=ax[0],
    fontsize=13,
    colors=['#34495E', '#566573', '#5D6D7E', '#85929E', '#AEB6BF', '#EBEDEF']
)
ax[0].set_title("Distribution of Wine Quality in Dataset", fontsize=20)
ax[0].legend(bbox_to_anchor=(1, 1), fontsize=12)

# Use sns.color_palette context
with sns.color_palette(['#EBEDEF', '#85929E', '#34495E', '#566573', '#5D6D7E', '#AEB6BF']):
    sns.countplot(
        x="Quality Category",
        data=df_category,
        ax=ax[1]
    )
ax[1].set_title("Frequency distribution of quality", fontsize=20)

# Annotate bars with percentages
for p in ax[1].patches:
    percentage = 100 * p.get_height() / len(df_category['Quality Category'])
    ax[1].annotate(f'{percentage:.1f}%', (p.get_x() + 0.1, p.get_height() + 0.1))

# Adjustments
plt.xticks(rotation=70, fontsize=12)
sns.set_context("paper", rc={"font.size":8, "axes.titlesize":20, "axes.labelsize":14})

plt.tight_layout()  # Adjust layout
plt.show()


quality_table= {'Values': [0, 1, 2, 3, 4, 5]}

df_comparison = pd.DataFrame.from_dict(quality_table).T
df_comparison.columns = ['Terrible', 'Very Poor', 'Poor', 'Good', 'Very Good', 'Excellent']

df_comparison.style.background_gradient(cmap='Blues')

sns.pairplot(df)
plt.show()

fig = plt.figure(figsize=(16, 8))

gs1 = gridspec.GridSpec(2, 6)
axs = []
for c, num in zip(df.columns, range(1, 12)):
    axs.append(fig.add_subplot(gs1[num - 1]))
    axs[-1].hist(df[c])
    plt.title(c)

plt.tight_layout()
plt.show()



plt.figure(figsize=(15, 8))

# Calculate absolute correlation with 'quality', sort, and exclude 'quality' itself
df_corr_bar = abs(df.corr()['quality']).sort_values()[:-1]

# Plot the bar plot with 'hue' and 'palette'
sns.barplot(
    x=df_corr_bar.index,
    y=df_corr_bar.values,
    palette="Blues_d",
    hue=df_corr_bar.index,
    dodge=False
).set_title('Feature Correlation Distribution According to Quality', fontsize=20)

# Remove the legend since it's unnecessary
plt.legend([],[], frameon=False)

# Adjust the x-axis labels
plt.xticks(rotation=70, fontsize=14)

# Show the plot
plt.show()

# Set the figure size globally for Seaborn plots
sns.set_theme(rc={'figure.figsize': (12, 9)})

# Create the boxplot with hue set to Quality Category
ax = sns.boxplot(
    x="Quality Category",
    y="volatile acidity",
    data=df_category,
    palette=['#34495E', '#566573', '#5D6D7E', '#85929E', '#AEB6BF', '#EBEDEF'],
    hue="Quality Category",  # Assigning the x variable to hue
    dodge=False              # Prevents separating bars for the same category
)

# Set titles and labels
ax.set_title('Boxplot for Volatile Acidity vs Quality', fontsize=20)
ax.set_xlabel('Quality', fontsize=14)
ax.set_ylabel('Volatile Acidity', fontsize=14)

# Hide the legend since it's unnecessary
plt.legend([], [], frameon=False)

# Show the plot
plt.show()


# Set the figure size globally for Seaborn plots
sns.set_theme(rc={'figure.figsize': (12, 9)})

# Create the boxplot with hue set to Quality Category
ax = sns.boxplot(
    x="Quality Category",
    y="alcohol",
    data=df_category,
    palette=['#34495E', '#566573', '#5D6D7E', '#85929E', '#AEB6BF', '#EBEDEF'],
    hue="Quality Category",  # Assigning the x variable to hue
    dodge=False              # Prevents separating bars for the same category
)

# Set titles and labels
ax.set_title('Boxplot for Alcohol vs Quality', fontsize=20)
ax.set_xlabel('Quality', fontsize=14)
ax.set_ylabel('Alcohol', fontsize=14)

# Hide the legend since it's unnecessary
plt.legend([], [], frameon=False)

# Show the plot
plt.show()

# Set the figure size
plt.figure(figsize=(22, 20))

# Create the regression plot
ax = sns.regplot(
    x="fixed acidity",
    y="citric acid",
    data=df,
    x_bins=25
)

# Set the limits for the x and y axes
plt.xlim(4, 16)
plt.ylim(0)

# Set titles and labels
ax.set_title('Correlation Between Fixed Acidity and Citric Acid', fontsize=20)
ax.set_xlabel('Fixed Acidity', fontsize=14)
ax.set_ylabel('Citric Acid', fontsize=14)

# Show the plot
plt.show()

# Set the figure size
plt.figure(figsize=(22, 20))

# Create the regression plot
ax = sns.regplot(
    x="fixed acidity",
    y="pH",
    data=df,
    x_bins=25
)

# Set titles and labels
ax.set_title('Correlation Between Fixed Acidity and pH', fontsize=20)
ax.set_xlabel('Fixed Acidity', fontsize=14)
ax.set_ylabel('pH', fontsize=14)

# Show the plot
plt.show()


# Group by 'quality' and calculate the mean for each group
average = df.groupby("quality").mean()

# Create a bar plot of the averages
average.plot(kind="bar", figsize=(20, 8))

# Set the title and labels
plt.title('Average Values by Quality', fontsize=20)
plt.xlabel('Quality', fontsize=14)
plt.ylabel('Average Value', fontsize=14)

# Show the plot
plt.show()

# Create a figure and axis
f, ax = plt.subplots(figsize=(10, 8))

# Extract the 'density' column directly
x = df['density']

# Plot the KDE with fill
sns.kdeplot(x, fill=True, color='blue', ax=ax)  # Use the existing ax for plotting

# Set the title for the plot
ax.set_title("Distribution of Density Variable", fontsize=20)

# Show the plot
plt.show()

# Create the figure with specified size
plt.figure(figsize=(15, 7))

# Plot sulphates vs Quality Category
sns.lineplot(data=df_category, x="Quality Category", y="sulphates", color="g", label="Sulphates", marker='o')
# Plot citric acid vs Quality Category
sns.lineplot(data=df_category, x="Quality Category", y="citric acid", color="b", label="Citric Acid", marker='o')

# Set the limits for the axes
plt.xlim(0, 5)  # Adjust if "Quality Category" has different values
plt.ylim(0)

# Set labels and title
plt.ylabel("Quantity", fontsize=14)
plt.xlabel("Quality", fontsize=14)
plt.title("Feature Impact on Quality", fontsize=20)

# Add a legend
plt.legend()

# Optionally add grid lines for better readability
plt.grid()

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()


