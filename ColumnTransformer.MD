ColumnTransformer in scikit-learn: A Beginner's Guide

A simple, hands-on introduction to ColumnTransformer, the scikit-learn tool that lets you apply different preprocessing steps to different columns of a dataset in one go.

Why ColumnTransformer?

Real datasets have mixed column types:

Column type	Example	What it needs
Numeric	age, salary	Scaling
Categorical	city, gender	Encoding
Not useful	id, name	Dropping

Applying each step manually is messy. ColumnTransformer handles all of it in a single object.

Analogy: think of it as a manager saying "do this task on these columns, and that task on those columns."

Syntax
python
ColumnTransformer([
    (name, transformer, [columns]),
    (name, transformer, [columns]),
])
Part	Meaning
name	A label for the step. It can be anything, but must be unique. It does not change the result.
transformer	The actual work: StandardScaler(), OneHotEncoder(), "passthrough", "drop"
[columns]	Which columns this step applies to
Example

Dataset:

name	height	weight	city
Ali	170	70	Lahore
Sara	160	55	Karachi
Ahmed	180	85	Lahore
Zara	165	60	Multan

Goal:

Scale height
One-hot encode city
Keep weight as it is
Drop name
python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

x = pd.DataFrame({
    "name": ["Ali", "Sara", "Ahmed", "Zara"],
    "height": [170, 160, 180, 165],
    "weight": [70, 55, 85, 60],
    "city": ["Lahore", "Karachi", "Lahore", "Multan"]
})

ct = ColumnTransformer([
    ("num", StandardScaler(), ["height"]),
    ("cat", OneHotEncoder(), ["city"]),
    ("keep", "passthrough", ["weight"]),
])

X_new = ct.fit_transform(x)
print(X_new)

Output:

[[ 0.169  0.     1.     0.    70.   ]
 [-1.183  1.     0.     0.    55.   ]
 [ 1.521  0.     1.     0.    85.   ]
 [-0.507  0.     0.     1.    60.   ]]
How to read the output

The output columns follow the order of the transformers you wrote:

Column	Meaning
1	Scaled height (from num)
2, 3, 4	One-hot city: Karachi, Lahore, Multan (from cat, alphabetical)
5	Original weight (from keep)
name is gone because it was not listed (default remainder="drop").
Each city row has exactly one 1, which marks that row's city.
Scaled values: 0 means average, positive means above average, negative means below average.

You can check the column names with:

python
print(ct.get_feature_names_out())
# ['num__height' 'cat__city_Karachi' 'cat__city_Lahore' 'cat__city_Multan' 'keep__weight']
The remainder parameter

What happens to columns you did not list?

Value	Behaviour
remainder="drop" (default)	Unlisted columns are removed
remainder="passthrough"	Unlisted columns are kept as they are, placed at the end
Shortcut: make_column_transformer

If you do not want to write names, scikit-learn generates them for you:

python
from sklearn.compose import make_column_transformer

ct = make_column_transformer(
    (StandardScaler(), ["height"]),
    (OneHotEncoder(), ["city"]),
    ("passthrough", ["weight"]),
)
Common mistakes
Numbers written as strings ('160' instead of 160): StandardScaler cannot scale text.
Missing commas between tuples or dictionary entries.
Typos in data or column names ("Karach" vs "Karachi"): column names must match the DataFrame exactly, including capital letters.
Listing the same column in two steps (for example scaling and passing through weight): it will appear twice in the output.
Duplicate step names: every name must be unique.
Forgetting to define X before calling fit_transform.
Key takeaways
ColumnTransformer applies different transformations to different columns.
Tuple structure: (name, transformer, [columns]).
The name is only a label; the transformer does the work.
Unlisted columns are dropped by default.
"passthrough" keeps a column untouched.
One-hot encoding turns a category into 0/1 "switches".
What's next
Using ColumnTransformer inside a Pipeline with a model
Handling missing values with SimpleImputer
handle_unknown="ignore" in OneHotEncoder
Requirements
pandas
scikit-learn
bash
pip install pandas scikit-learn
