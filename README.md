[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MattJBritton/ForestfortheTrees/master?urlpath=lab/tree/notebook.ipynb)

# ForestfortheTrees
This library provides utilities for generating visual explanations of Gradient Boosting models. I recommend you jump in through the Binder link above, which renders the notebook.ipynb file. This interactive Jupyter notebook is an [Explainable](https://explorabl.es/) that showcases the value of the library and provides sample code.

## Installation

Alternatively, you can run the notebook locally by cloning the repository and then performing the following:

1. Navigate into the package directory. ```cd ForestForTheTrees```
2. Install the conda environment. ```conda env create binder/environment.yml```
3. Activate the conda environment. ```conda activate ForestForTheTrees```
4. Run the postBuild script (this installs the appropriate jupyterlab extension required to display interactive widgets) ```bash binder/postBuild``` or just run this command directly ```jupyter labextension install @jupyter-widgets/jupyterlab-manager```.
5. Fire up Jupyter Lab, run all cells and begin interacting with the notebook. ```jupyter lab notebook.ipynb```

Note that a recent version of Jupyter Lab (included in the environment) is required to run this notebook - Jupyter notebooks will not work (at least out of the box). This is due to some peculiarities in the interaction of Altair, ipywidgets, and Jupyter. 

I recommend running all cells as soon as the notebook is opened. Due to the nature of the interactive widgets, it is not possible to save the state, so the notebook is saved without output. If you are perusing the full document, each cell will have run by the time you get to it. This applies whether viewing locally or via Binder. 

## Usage
As mentioned above, the best way to get a sense of how Gradient Boosting models can be explained with ForestForTheTrees is to run the Binder link above. To get started quickly, adapt the minimal example below:

```python
#load dataset
dataset_df = pd.read_csv("Some_file.csv")
target_column = "Target"  #the value to predict

#build model
model = GradientBoostingRegressor(
    num_estimators = 100
)

#fit model
model.fit(
    dataset_df.drop(target_column, axis = 1),
    dataset.loc[:,target_column]
) #you should build a good model here using train/test split

#initialize ForestForTheTrees with dataset, model, and target
f2t = ft.ForestForTheTrees(
    dataset = dataset_df, #pass bike instead to use the sample dataset
    model = None,
    target_col = "Ridership"
)

#extract the underlying structure of the model
#this must be called before displaying the visual explanation
f2t.extract_components()

#output the visual explanation at the selected fidelity
f2t.explain(
    fidelity_threshold = .95
)
```
![5-chart explanation for bike dataset](https://github.com/MattJBritton/ForestfortheTrees/blob/master/readme_resources/5_chart_explanation.png "5 chart explanation for bike dataset")

## Development
This library is under active development - please review the [Issues](https://github.com/MattJBritton/ForestfortheTrees/issues) tab for current priorities. Feature requests and bug reports are welcomed! If you find this library useful, please feel free to message me and let me know how it went. 

Developed using Python and the Python data science stack, particularly [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/), and [scikit-learn](https://scikit-learn.org/stable/). [Altair](https://altair-viz.github.io/) was used for data visualization.
