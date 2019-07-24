## Description: 

## Requirements:
You can run DogR using python-3. The required packages are as follow: 
* `numpy`
* `scipy`
* `pandas`
* `absl-py`
* `sklearn`
* `statsmodels`
* `matplotlib`
* `seaborn`

There are three folders: `\Data\`, `\Figures\` and `\Pickels\`. The `.csv` file of the dataset needed to be in `\Data\` folder. The parameters of the learned model for each dataset, is going to be save in a subfolder (with the name of the dataset) in `\Pickels\` folder. The generated figures are going to be store in `\Figures\` folder. 

## How to run

### Prediction
For running the prediction task on your data, you need to specify name of dataset by `--dataset`, the y variable by `--y_var`, minimum and maximum number of components to consider as hyper parameter tuning by `--min_comp` and `--max_comp`. For example, following command would run the prediction task for Tract Data, by considering y variables as `meanV` and number of components in range `1-7`. 

`python prediction.py --dataset=Tract_Data --y_var=meanV --min_comp=1 --max_comp=7`

### Model Data
The parameters of DogR model could be learned for specific data using the following command. 

`python model_data.py --dataset=Tract_Data --y_var=meanV --num_com=5`

where `--num_com` is number of components. After learning the parameters of DogR model for the dataset, the parameters is going to be save in `\Pickels\dataset\` folder, to be used lated for analyzing the model. 

### Best number of components 
To find the best number of components you can use generated AIC-BIC plots, by running the following command:

`python number_of_components.py --dataset=Tract_Data --y_var=meanV`

which is going to generate a plot in `\Figures\dataset\` folder, for AIC-BIC value of the model using different number of components. The range of number of components is `1-10`. 

### Analyze 
You can analyze the learned model, by running the following command:

`python analyze_model.py --dataset=Tract_Data --y_var=meanV`

The learned parameters stored in `\Pickels\dataset\` folder is going to be used for generating plots and reporting statistical tests. The generated plots is going to be save in `\Figures\dataset\` folder. 
