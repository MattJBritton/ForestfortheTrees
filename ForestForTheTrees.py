#imports
#from __future__ import division
import time
import datetime
import copy
from itertools import product
import operator
from collections import OrderedDict
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

import altair as alt
alt.renderers.enable("default")
from IPython.display import display

#seed for reproducibility
np.random.seed = 15

class ForestForTheTrees:
    
    ALLOWED_REGRESSION_ERROR_METRICS = ["r_squared", "mae"]  
        
    def __init__(
        self, 
        dataset, 
        model, 
        target_col, 
        task = "Regression", 
        sample_size = None, 
        num_tiles = 40, 
        quantiles = False
    ):
        
        """ Initialize an instance with a dataset, model, target, task, and other parameters
        
            Generates an internal data representation from the dataframe, fits the model if not done already,
            initializes default variables, and raises errors if bad arguments are passed.
            
            --Arguments
            dataset (pd.DataFrame or string):
                If DataFrame, the whole dataset with predictor and target features
                If String, the name of one of the sample datasets in get_sample_dataset
            model (sklearn.GradientBoostingRegressor or None)
                Model to explain. If None, initialize and fit model with default hyperparameters
            target_col (String):
                Column name found in dataset.columns. Value to be predicted
            task (String):
                "Regression" is the only task currently supported
            sample_size (Integer or None):
                For large datasets, select a number of datapoints to randomly sample (without replacement)
                when evaluating explanation components. If none, use the whole dataset.
            num_tiles (Integer):
                For quantitative features, the number of bins into which the feature range is divided,
                and therefore also the number of squares in the heatmap for that feature.
                Bounds the max size of a single component heatmap at num_tiles^2.
                Higher numbers result in more accurate explanations at the cost of slower running time.
            quantiles (Boolean):
                If true, divide feature range into num_tiles bins of equal numbers of datapoints, rather than equal size.
                
            --Returns
            None
        """
        
        #initialize variables
        self.model = model
        self.target_col = target_col
        self.task = task
        self.mean_prediction = None
        self.no_predictor_features = []
        self.oned_features = []   
        self.binned_data = None
        self.sample_size = sample_size #may be set to none here, will be handled in load_dataset()
        self.num_tiles = num_tiles
        self.quantiles = quantiles
        self.predictions_base = None
        self.chart_components = {}
        self.explanation_components = {}
        self.base_explanation = []
        self.evaluation_details = []
        self.base_components = []
        self.explanation = []
        self.cache = {}

        #get sample dataset if string passed
        if isinstance(dataset, str):
            dataset = ForestForTheTrees.get_sample_dataset(dataset)
        elif not isinstance(dataset, pd.DataFrame):
            raise ValueError(f"Invalid type {type(dataset)} passed for dataset argument. Please provide "\
                            + "a DataFrame or a string if using one of the sample datasets.")
        
        #generate internal data representation
        features = [x for x in dataset.columns if x != self.target_col]
        self.x = dataset.loc[:,features].values
        self.y = dataset.loc[:,target_col]
        self.feature_names = features
        self.feature_locs = {x:i for i,x in enumerate(features)}
        self.feature_ranges = {
            feature : self._get_quantiles(feature)
            for feature in self.feature_names
        } 
        if self.sample_size is None: #if no sample size use whole dataset
            self.sample_size = self.x.shape[0]
        self.binned_data = self._bin_data() 
        
        #if no model passed, instantiate one with default settings
        if self.model is None:
            if self.task == "Regression":
                self.model = GradientBoostingRegressor(
                    n_estimators = 300, 
                    max_depth = 2, 
                    learning_rate = 1.
                )                
            else:
                raise NotImplementedError("Regression is the only task currently implemented.")
                
        #now handle models which are not yet fitted
        #the ideal scenario is that models are passed already fitted according to appropriate train/test split
        #so this is provided primarily as a convenience
        try:
            if check_is_fitted(self.model, "tree_", "OK") == "OK":
                pass
            else:
                self.model.fit(self.x, self.y)
        except NotFittedError:
            self.model.fit(self.x, self.y)
        
        #all models should be fitted at this point
        self.pred_y = self.model.predict(self.x)
    
    def get_sample_dataset(dataset_name):
        
        """Return dataframe of sample dataset"""
        
        if dataset_name != "bike":
            raise NotImplementedError("The only dataset currently available is the bike dataset.")
        
        if dataset_name == "bike":
            def _datestr_to_timestamp(s):
                return time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d").timetuple())

            dataLoad = pd.read_csv('data/bike.csv')
            dataLoad['dteday'] = dataLoad['dteday'].apply(_datestr_to_timestamp)
            dataLoad = pd.get_dummies(dataLoad, prefix=["weathersit"], columns=["weathersit"], drop_first = False)

            #de-normalize data to produce human-readable features.
            #Original range info from http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
            dataLoad["hum"] = dataLoad["hum"].apply(lambda x: x*100.)
            dataLoad["windspeed"] = dataLoad["windspeed"].apply(lambda x: x*67.)
            #convert Celsius to Fahrenheit
            dataLoad["temp"] = dataLoad["temp"].apply(lambda x: (x*47. - 8)*9/5 +32)
            dataLoad["atemp"] = dataLoad["atemp"].apply(lambda x: (x*66. - 16)*9/5 + 32)

            #rename features to make them interpretable for novice users
            feature_names_dict = {
                "yr":"First or Second Year", 
                "season":"Season", 
                "hr":"Hour of Day", 
                "workingday":"Work Day",
                "weathersit_2":"Misty Weather",
                "weathersit_3":"Light Precipitation",
                "weathersit_4":"Heavy Precipitation",
                "temp":"Temperature (F)",
                "atemp":"Feels Like (F)",
                "hum":"Humidity",
                "windspeed":"Wind Speed"
            }
            dataLoad = dataLoad.rename({"cnt" : "Ridership"}, axis = 1)
            dataLoad = dataLoad.rename(mapper=feature_names_dict,axis=1) 
            return dataLoad.loc[:, list(feature_names_dict.values()) + ["Ridership"]]
        
    def get_series_as_array(self, ser):
        return np.array(ser).reshape(-1,1)
        
    def return_dataset_as_dataframe(self):
        """Construct dataframe from internal NumPy data representation"""
        return pd.DataFrame(data = np.hstack((self.x, self.get_series_as_array(self.y))),
                            columns = self.feature_names + [self.target_col] 
                           )

    def _bin_data(self):
    
        """For each pair of features in the dataset, generate an array of bin values in an X/Y grid."""
    
        prediction_contributions = {}
        sample_data = pd.DataFrame(
            self._get_sample(self.x),
            columns = self.feature_names
        )
        for key in self._get_feature_pairs():
            tempH = np.digitize(
                sample_data.loc[:,key[0]],
                self.feature_ranges[key[0]]
            )-1.
            tempV = np.digitize(
                sample_data.loc[:,key[1]],
                self.feature_ranges[key[1]]
            )-1.
            prediction_contributions[key] = (tempV*len(self.feature_ranges[key[0]]) + tempH).astype(int)
        return prediction_contributions            
        
    def _get_coordinate_matrix(self, lst, length, direction):
        """Expand a list out to length to create the indices for a 2d feature array"""
        if direction=="h":
            return lst*length
        else:
            return [item for item in lst\
             for i in range(length)]   

    def _get_quantile_matrix(self, feat1, feat2):
        """Get the flattened lists of indices needed to index a 2d array of 2 features"""
        h = self._get_coordinate_matrix(
            list(self.feature_ranges[feat1]),
            len(self.feature_ranges[feat2]),
            "h"
        )
        v = self._get_coordinate_matrix(
            list(self.feature_ranges[feat2]),
            len(self.feature_ranges[feat1]),
            "v"
        )                      
        return h,v 

    def _get_leaf_value(self, tree, node_position):
        """Access one node in a tree by its position"""
        return tree.value[node_position]       

    def _get_feature_pair_key(self, feat1, feat2):
        """Generate a tuple key from two features with stable sorting. Used to define chart components."""
        if self.feature_ranges[feat1].shape[0] == self.feature_ranges[feat2].shape[0]:
            #need stable order so keys with same number of quantiles appear in only one order
            return tuple(sorted([feat1, feat2]))
        elif self.feature_ranges[feat1].shape[0] > self.feature_ranges[feat2].shape[0]:
            return tuple([feat1, feat2])
        else:
            return tuple([feat2, feat1])        

    def _get_quantiles(self, feat):
        """For a given feature, generate the unique categories or quantitative bins"""
        loc = self.feature_locs[feat]
        if np.unique(self.x[:,loc]).shape[0] < 30 or type(self.x[0,loc]) is str: #is categorical/ordinal?
            return np.unique(self.x[:,loc])
        else:
            if self.quantiles:
                return np.around(
                    np.unique(
                        np.quantile(
                            a=self.x[:,loc],
                            q=np.linspace(0, 1, self.num_tiles)
                        )
                    ),
                    1)
            else:
                return np.around(
                    np.linspace(
                        np.min(self.x[:,loc]), 
                        np.max(self.x[:,loc]),
                        self.num_tiles
                    )
                    ,1)  
            
    def _reduce_to_1d(self, arr, threshold, direction):
        """Determine if the loss from reducing an array to its first column or row is below user-defined threshold"""
        if direction == "h":
            reduced_arr = arr - arr[:,0].reshape(-1,1)
        else:
            reduced_arr = arr - arr[0,:].reshape(1,-1)
        return (np.max(np.abs(reduced_arr))/np.max(np.abs(arr))) <= threshold               
        
    def _get_sample(self, arr):
        """Get the first self.sample_size datapoints in the dataset as a sample"""
        #NOTE: this needs to be improved to use a persistent random sample of datapoint ids instead
        #which should be set upon initialization or whenever self.sample_size is changed
        return arr[:self.sample_size]
    
    def _get_predictions_base(self):
        """Generate the default (estimators = 0) predictions consisting of the target mean in an array of sample_size."""
        return np.full((self.sample_size,1), np.mean(self.y))
    
    def _get_empty_sample(self, size = None):
        """Get an array of zeroes of user-defined size."""
        return np.full((self.sample_size if size is None else size,1), 0)
    
    def _get_explanation_accuracy(self, explanation_predictions, error_metric, subset_indices = None):
        """ Compare a list of predictions to the actual Y value and generate desired error score
            
            --Arguments
            explanation_predictions (np.array):
                Array of same length and ordering as self.y
            error_metric (string)
                Name of score to apply. Currently only r_squared and mae are implemented
                
            --Returns
            error_metric: float
                Score from desired error function
        """
        if error_metric == "r_squared":
            func = r2_score
        elif error_metric == "mae":
            func = mean_absolute_error
            
        if subset_indices is not None:
            return func(self.pred_y[subset_indices], explanation_predictions[subset_indices])
        else:
            return func(self._get_sample(self.pred_y), explanation_predictions)
        
    def _get_prediction_contributions(self, chart, data_positions):
        """ For a given chart/explanation component, get the prediction contribution corresponding to the bin in which a datapoint falls.
            
            --Arguments
            chart: np.array:
                A chart component or explanation component.
                This has shape of (len(_get_quantiles(feat1)), len(_get_quantiles(feat2)))
            data_positions: np.array
                Array of positions to "take" from chart components, each position corresponding to the binned
                feat1+feat2 values for one datapoint. Positions are flattened into a single value.
                
            --Returns
            values: np.array
                Array in which the value of each iten corresponds to the prediction contribution that chart
                made for the datapoint represented at position.
        """        
        return np.take(chart, data_positions)
    
    def _sum_arrays(self, temp_outputs, keyMain, keyAdd, arr_to_add):
        """ Roll up an aggregated view of a chart component into a more predictive one.
            
            --Arguments
            temp_outputs: dict of dict of arrays:
                Dictionary of feature pair tuples, each with multiple outputs generated by X
            keyMain: tuple(feat1, feat2)
                The key of the chart to be modified
            keyAdd: tuple(feat1, feat2)
                The key of the chart being 'rolled up'
            arr_to_add: string
                The particular form of derived output to roll up, based on which feature axis is being aggregated
                and whether the other axis was already rolled up into another chart
                
            --Returns
            values: np.array
                The updated version of temp_outputs[keyMain]['output']
        """          
        
        return temp_outputs[keyMain]["output"]\
    + temp_outputs[keyAdd][arr_to_add].reshape(
            temp_outputs[keyMain]["output"].shape[0]
            if(keyMain[1]==keyAdd[1] or keyMain[1]==keyAdd[0])
            else 1,-1
        )
    
    def _drop_alternate_outputs(self,component):
        """Discard output_v, output_h, etc once they have been folded into 'output' during extract_components()"""
        return {"output": component["output"]}
    
    def _get_prediction_contributions_df(self, components, explanation):
        """Convert the prediction contributions for a set of components into a dataframe"""
        return np.hstack(
            tuple(
                [
                    self._get_prediction_contributions(
                        components[expKey]["output"],
                        self.binned_data[expKey]
                    ).reshape(-1, 1)\
                    for expKey in explanation                    
                ]
            )
        )
    
    def get_prediction_contributions_by_key(self, components, explanation):
        """ Generate prediction contributions by components of an explanation
        for each datapoint in get_sample(self.x)
            
            --Arguments
            components: dict of dict of arrays:
                Typically, self.chart_components or self.explanation_components
            explanation: list of feature pair tuples, ordered by importance
                The key of the chart to be modified
                
            --Returns
            values: dict of arrays
                Dict of feature pair tuples where each value is the point-wise prediction contributions
                for each datapoint in get_sample(self.x)
        """
        return {
            expKey : 
            self._get_prediction_contributions(
                components[expKey]["output"],
                self.binned_data[expKey]
            ) for expKey in explanation
        }
    
    def evaluate_explanation(self, error_metric = "r_squared"):
        """Evaluate an explanation's fidelity to self.model using the provided error metric
            
            --Arguments
            error_metric: String
                The error metric by which to evaluate the explanation's fidelity
                to the model. Currently, only "r_squared" is accepted.
                
            --Returns
            fidelity: Float
                Depends on the definition of the error metric. R-squared is a float from -inf => 1.
        """        
        if self.task == "Regression":
            if error_metric in self.ALLOWED_REGRESSION_ERROR_METRICS: 
                return self._evaluate_single_explanation(self.chart_components, self.explanation, error_metric)
            else:
                raise ValueError(f"Not an allowed error metric for {self.task}. Try one of"\
                + f"{self.ALLOWED_REGRESSION_ERROR_METRICS}.")
        elif self.task == "Classification":
            raise NotImplementedError("Classification not yet implemented.")

    def _evaluate_single_explanation(self, components, explanation, error_metric, subset_indices = None):
        """Internal version of evaluate_explanation()
            
            --Arguments
            components: dict of dict of arrays
                    Dictionary of feature pair keys with each value having output, output_v, output_h, etc
                    Each form of output is an array of prediction contributions indexed by chart_indices              
            explanation: List
                List of feature pair keys specifying an explanation. Typically from self.explanation or self.base_explanation
            error_metric: String
                The error metric by which to evaluate the explanation's fidelity
                to the model. Currently, only "r_squared" is accepted.
                
            --Returns
            fidelity: Float
                Depends on the definition of the error metric. R-squared is a float where -inf < x <= 1.
        """        
        
        return self._get_explanation_accuracy(
            self.predictions_base +\
            np.sum(
                np.array(
                    list(
                        self.get_prediction_contributions_by_key(
                            components,
                            explanation
                        ).values()
                    )
                ), 
                axis = 0
            ).reshape(-1,1),
            error_metric,
            subset_indices
        )
    
    def _get_parallel_coordinate_columns(self, explanation, cumulative):
        
        """Get the list of column names for a parallel coordinates chart showing prediction contributions
            
            --Arguments           
            explanation: List
                List of feature pair keys specifying an explanation. Typically from self.explanation or self.base_explanation
            cumulative: Boolean
                Whether the parallel coordinates chart calling this displays cumulative predictions or individual 
                prediction components. Controls whether "mean y" and "prediction" are added to the output
                
            --Returns
            column_names: List
                List of stringified column names for Altair to use in the chart generated by visualize_datapoints()
        """         
        
        #make these strings because Altair doesn't like a tuple as a key and turns it into a list
        return (["mean y"] if cumulative else [])\
    + [x[0] + "," + x[1] for x in explanation]\
    + (["prediction"] if cumulative else [])
    
    def _get_altair_data_type(self, feature_name, abbreviation = True):
        
        """Return the data type of a field in the format required by Altair field encodings
            
            --Arguments           
            feature_name: String
                Name of a valid feature in this dataset.
            abbreviation: Boolean
                Whether to return the data type as the full Altair name or the abbreviated form
                
            --Returns
            data_type: String
                Name or abbreviation of the feature's data type
        """          
        if feature_name not in self.feature_ranges:
            raise ValueError("Feature is not present in dataset. Check f2t.feature_ranges()")
        elif self.feature_ranges[feature_name].shape[0] == self.num_tiles:
            return "Q" if abbreviation else "quantitative"
        else:
            return "O" if abbreviation else "ordinal"    
    
    def _get_datapoint_contributions(self, components, explanation):
        
        """Build the data structure for the chart generated by visualize_datapoints().
        This is basically the result of _get_prediction_contributions_df() with additional columns
        for prediction loss, sorting, etc. 
            
            --Arguments
            components: dict of dict of arrays
                    Dictionary of feature pair keys with each value having output, output_v, output_h, etc
                    Each form of output is an array of prediction contributions indexed by chart_indices              
            explanation: List
                List of feature pair keys specifying an explanation. Typically from self.explanation or self.base_explanation
                
            --Returns
            datapoints: pandas.DataFrame
                Dataframe with one row per datapoint, and columns for each feature pair key indicating their contribution
                to the prediction, with additional columns for sort order, prediction, mean_y, and explanation_loss.
        """ 
        
        contributions = self._get_prediction_contributions_df(components, explanation)
        #raw contributions
        arr = np.hstack(
            (
                self._get_empty_sample().reshape(-1,1),
                contributions.reshape(self.sample_size, -1),
                self._get_sample(self.pred_y).reshape(-1,1),
                np.array([0. for x in range(self.sample_size)]).reshape(-1,1)
            )
        )
        
        #cumulative version
        arr_cum = np.cumsum(
            np.hstack(
                (
                    self._get_predictions_base().reshape(-1,1),
                    contributions.reshape(self.sample_size,-1)
                )
            ),
            axis = 1
        )
        
        arr_cum = np.hstack(
            (
                arr_cum,
                self._get_sample(self.pred_y).reshape(-1,1),
                np.array([1. for x in range(self.sample_size)]).reshape(-1,1)
            )
        )

        arr_df = pd.DataFrame(
            arr,
            columns = self._get_parallel_coordinate_columns(explanation, True) + ["view"]
        )
        
        #combine arrays
        arr_cum_df = pd.DataFrame(
            arr_cum,
            columns = self._get_parallel_coordinate_columns(explanation, True) + ["view"]
        )
        
        #generate datapoint id columns
        arr_df = arr_df.reset_index(drop = False)
        arr_cum_df = arr_cum_df.reset_index(drop = False)
        datapoints = pd.concat([arr_df, arr_cum_df])
        
        #couldn't do this earlier, as you can't vstack a mixed type array
        datapoints["view"] = datapoints["view"].apply(lambda x:\
                                                      'Predictions by Chart'\
                                                      if x < 1. else\
                                                      'Cumulative Predictions'\
                                                     )
        #calculate explanation loss
        datapoints["explanation_loss"] = np.abs( #otherwise the chart is hard to read with 0 in the middle of the axis
            datapoints.loc[:,"prediction"] - datapoints.iloc[:,-3]#last cumulative column
        )
        
        datapoints["prediction_index"] = datapoints["prediction"]
        datapoints = datapoints.melt(
            id_vars = ['index', "prediction_index", "view", "explanation_loss"],
            var_name = 'component',
            value_name = 'contribution'
        )
        
        #rename prediction again
        datapoints = datapoints.rename({"prediction_index" : "prediction"}, axis = 1)

        #drop fake columns for "Predictions by Chart"
        datapoints = datapoints[
            (datapoints["view"] == "Cumulative Predictions")
            | (~datapoints["component"].isin(["prediction", "mean y", "explanation_loss"]))
        ]
        
        #build sort column for Altair
        datapoints["sort"] = datapoints["component"].apply(lambda x:
                                                           self._get_parallel_coordinate_columns(
                                                               explanation,
                                                               True
                                                           ).index(x)
                                                          )   
        return datapoints

    def _copy_chart_components(self):
        """Generate a deep copy of self.chart_components so they can be safely modified during roll-up"""
        return copy.deepcopy(self.chart_components)  
    
    def _get_feature_pairs(self):
        """Generate the full list of feature pair keys based on the feature names of the dataset"""
        return [
            self._get_feature_pair_key(key[0], key[1])
            for key in [tuple(t) for t in product(self.feature_names, repeat = 2)]
        ]       

    def _rollup_components(self, explanation):
        """ROLLUP is under development and may not function as intended"""
        temp_outputs = self._copy_chart_components()
        for keyRollup in [k for k in self.chart_components.keys() if k not in explanation]:
            hUsed = False
            vUsed = False
            for keyExisting in explanation:
                if (keyRollup[1] == keyExisting[0] or keyRollup[1] == keyExisting[1]) and not hUsed:
                    hUsed = True
                    if vUsed:
                        temp_outputs[keyExisting]["output"] = self._sum_arrays(
                            temp_outputs,
                            keyExisting, 
                            keyRollup,
                            "output_HReduced"
                        )
                        break
                    else:
                        temp_outputs[keyExisting]["output"] = self._sum_arrays(
                            temp_outputs,
                            keyExisting, 
                            keyRollup,
                            "output_H"
                        )                           
                elif (keyRollup[0] == keyExisting[0] or keyRollup[0] == keyExisting[1]) and not vUsed:
                    vUsed = True
                    if hUsed:
                        temp_outputs[keyExisting]["output"] = self._sum_arrays(
                            temp_outputs,
                            keyExisting, 
                            keyRollup,
                            "output_VReduced"
                        )                          
                        break
                    else:
                        temp_outputs[keyExisting]["output"] = self._sum_arrays(
                            temp_outputs,
                            keyExisting, 
                            keyRollup,
                            "output_V"
                        )  
        return temp_outputs   
    
    def visualize_estimator(self, estimator_nums, try_collapse = False,
                            print_function_text = True, auto_display = True):
        
        """Visualize the chart components for 1 or more individual estimators within a model.
            
            --Arguments
            estimator_nums: Integer or List
                The list of estimators to visualize. Represents the index of the tree in model.estimators_
            try_collapse: Boolean
                If true, collapse as many of the estimators if possible, if they share the same feature pair key.
            print_function_text: Boolean
                If true, print the text of the decision function specified by the tree on top of the chart.
            auto_display: Boolean
                If true, display the chart. Otherwise, just return it for manual alteration and display   
                
            --Returns
            output: Altair.chart
                Returns the Altair visualization spec for further modification if necessary.
                If auto_display = True, this can be discarded.
        """         
        
        estimators = [self.model.estimators_[estimator_nums]] if type(estimator_nums) is int\
        else [x for i,x in enumerate(self.model.estimators_) if i in estimator_nums] 
        
        chart_components,\
        chart_indices,\
        _,\
        _,\
        function_texts = self._extract_components(
            try_collapse, 
            estimators,
            print_function_text
        )
        
        if print_function_text:
            if len(function_texts) == 1:
                for x in [x for x in list(function_texts.values())[0]["function_texts"]]:
                    print(x)
            else:
                for x in [x["function_texts"] for x in list(function_texts.values())]:
                    for text in x:
                        print(x[0])
        chart = self._visualize_components(
            chart_components.keys(),
            chart_components,
            chart_indices,
            None, 
            None,
            None,
            300,
            4
        )
        if auto_display:
            display(chart)
        return chart
    
    def _get_function_text(self, decision_func_dict):
        
        """Convert the internal representation of a tree into a natural language if-then statement of the decision function.
            
            --Arguments
            decision_func_dict: Dict
                The representation of an estimator used in _extract_components(). See that function for more details.
                
            --Returns
            text: String
                Decision function for the passed estimator as an if-then statement.
        """          
        
        def _get_left_right_text(op, le, gt):
            if op == operator.le:
                return " is less than or equal to ", str(round(le,1)), str(round(gt,1))
            else:
                return " is greater than ", str(gt), str(le)
                
        
        if "feature_name" in decision_func_dict: #1-deep
            
            comparison_text, left, right = _get_left_right_text(
                decision_func_dict["operator"],
                decision_func_dict["prob_le"],
                decision_func_dict["prob_gt"]
            )
            
            text = "If " + decision_func_dict["feature_name"] + comparison_text\
            + str(round(decision_func_dict["threshold"],1)) + " then " + left\
            + " else " + right + ". "
            
            return text
        
        else: #2-deep
            comparison_text_1, left_1, right_1 = _get_left_right_text(
                decision_func_dict["operator_1"],
                0,
                0
            )

            comparison_text_2, left_2, right_2 = _get_left_right_text(
                decision_func_dict["operator_2"],
                decision_func_dict["prob_le"],
                decision_func_dict["prob_gt"]
            ) 
            
            
            text = "If " + decision_func_dict["feature_name_1"] + comparison_text_1\
            + str(round(decision_func_dict["threshold_1"],1)) + " then proceed. If "\
            + decision_func_dict["feature_name_2"] + comparison_text_2\
            + str(round(decision_func_dict["threshold_2"],1)) + " then " + left_2 + " else " + right_2 + ". "
            
            return text
    
    def extract_components(self, collapse_1d = True, return_text = False):
        
        """ Extract chart components from the underlying tree structures.
            
            --Arguments
            collapse_1d: Boolean
                If True, perform basic rollup and add 1d charts to the chart component
                that has the highest feature importance among components that include the field in question
            return_text: Boolean
                If True, generate text descriptions of the function described by each tree.
                Generally only necessary for instructional purposes, see notebook.ipynb.
                
            --Returns
            None
                Chart components, chart indices, and debugging data saved to internal state
                See _extract_components() for more detail.
        """          
        
        self.chart_components,\
        self.chart_indices,\
        self.no_predictor_features,\
        self.oned_features,\
        self.estimator_texts = self._extract_components(collapse_1d, self.model.estimators_, return_text)
        
        self.predictions_base = self._get_predictions_base()
        
        #get the full explanation and store it. one reason for this is so that
        #charts can also be sorted appropriately
        #don't save explanation components as they will be the same as chart_components
        self.base_explanation, _, _\
        = self._explain(1., None)        
        
    def _extract_components(self, collapse_1d, estimators, return_text):

        """ Extract chart components from the underlying tree structures.
        Internal version of extract_components()
            
            --Arguments
            collapse_1d: Boolean
                If True, perform basic rollup and add 1d charts to the chart component
                that has the highest feature importance among components that include the field in question
            return_text: Boolean
                If True, generate text descriptions of the function described by each tree.
                Generally only necessary for instructional purposes, see notebook.ipynb.
                
            --Returns
                chart_components: dict of dict of arrays
                    Dictionary of feature pair keys with each value having output, output_v, output_h, etc
                    Each form of output is an array of prediction contributions indexed by chart_indices
                chart_indices: dict of dict of lists
                    Dictionary of feature pair keys with each value having "h_indices" and "v_indices",
                    each of which is a list containing the bins for the feature in the horizontal or vertical position
                no_predictor_features: list of feature pair keys
                    List of feature tuples for those pairs that were not included in any trees in the model
                self.oned_features: list of features
                    List of features (tuple of form (feat1, feat1)) that were rolled up
                    If collapse_1d is False, return empty list
                self.estimator_texts: list of strings
                    If return_text, the string at position i in the list represents the decision function
                    for tree i in self.mode. If False, return empty list.
        """         
        
        #generate data structure for pairwise charts
        feature_pairs = {
            key : {
                "map":None,
                "predicates":[],
                "function_texts":[]
            }
            for key in self._get_feature_pairs()
        }      

        for key, value in feature_pairs.items():
            h, v = self._get_quantile_matrix(key[0], key[1])
            value["map"] = np.array(
                [
                    {
                        key[0] : x,
                        key[1] : y
                    }
                    for x,y in zip(h,v)
                ]
            ).reshape(len(self.feature_ranges[key[1]]), len(self.feature_ranges[key[0]]))

        for modelT in estimators:
            curr_model = modelT[0]
            feature_ids = {
                i : {
                    "number":x,
                    "name":self.feature_names[x]
                } for i,x in enumerate(list(curr_model.tree_.feature))
                if x >= 0
            } #-2 means leaf node

            #for 1-layer trees
            if curr_model.tree_.feature[1] < 0:
                feature_pair_key = self._get_feature_pair_key(
                    feature_ids[0]["name"],
                    feature_ids[0]["name"]
                )
                decision_func_dict = {
                    "feature_name": feature_ids[0]["name"],
                    "threshold": curr_model.tree_.threshold[0],
                    "operator": operator.le,
                    "prob_le": self._get_leaf_value(curr_model.tree_, 1),
                    "prob_gt": self._get_leaf_value(curr_model.tree_, 2)
                }       
                #build the predictive function used in the decision tree
                def dt_predicate(data_case, decision_func_dict=decision_func_dict):
                    if decision_func_dict["operator"](\
                                                        data_case[decision_func_dict["feature_name"]],\
                                                        decision_func_dict["threshold"]\
                                                       ):
                        return decision_func_dict["prob_le"]
                    else:
                        return decision_func_dict["prob_gt"]        
            else:
                for node_position in [1,4]: #positions for left and right nodes at layer 2
                    if node_position in feature_ids:
                        feature_pair_key = self._get_feature_pair_key(
                            feature_ids[0]["name"], 
                            feature_ids[node_position]["name"]
                        )
                        #get the decision rules
                        decision_func_dict = {
                            "feature_name_1": feature_ids[0]["name"],
                            "threshold_1": curr_model.tree_.threshold[0],
                            "operator_1": operator.le if node_position == 1 else operator.gt,

                            "feature_name_2": feature_ids[node_position]["name"],
                            "threshold_2": curr_model.tree_.threshold[node_position],
                            "operator_2": operator.le,

                            "prob_le": self._get_leaf_value(curr_model.tree_, node_position+1),
                            "prob_gt": self._get_leaf_value(curr_model.tree_, node_position+2)
                        }
                        #build the predictive function used in the decision tree
                        def dt_predicate(data_case, decision_func_dict=decision_func_dict):
                            if decision_func_dict["operator_1"](\
                                                                data_case[decision_func_dict["feature_name_1"]],\
                                                                decision_func_dict["threshold_1"]\
                                                               ):
                                if decision_func_dict["operator_2"](\
                                                                    data_case[decision_func_dict["feature_name_2"]],\
                                                                    decision_func_dict["threshold_2"]\
                                                                   ):
                                    return decision_func_dict["prob_le"]
                                else:
                                    return decision_func_dict["prob_gt"]
                            else:
                                return 0.

                    else: #asymmetric tree, this is a leaf node
                        feature_pair_key = self._get_feature_pair_key(
                            feature_ids[0]["name"], 
                            feature_ids[0]["name"]
                        )
                        decision_func_dict = {
                            "feature_name": feature_ids[0]["name"],
                            "threshold": curr_model.tree_.threshold[0],
                            "operator": operator.le if node_position == 1 else operator.gt,
                            "prob": curr_model.tree_.value[node_position]
                        }
                        #build the predictive function used in the decision tree
                        def dt_predicate(data_case, decision_func_dict=decision_func_dict):
                            if decision_func_dict["operator"](\
                                                                data_case[decision_func_dict["feature_name"]],\
                                                                decision_func_dict["threshold"]\
                                                               ):
                                return decision_func_dict["prob"]
                            else:                         
                                return 0.                 

                    feature_pairs[feature_pair_key]["predicates"].append(dt_predicate)
                    if return_text:
                        feature_pairs[feature_pair_key]["function_texts"].append(
                            self._get_function_text(
                                decision_func_dict
                            )
                        )

        #now calculate output array for each feature pair
        for key, value in feature_pairs.items():
            arrs = []
            for predicate in value["predicates"]:
                f = np.vectorize(predicate)
                arrs.append(f(value["map"]))
            if len(arrs) > 0:
                #details of vote aggreggation method for random forest
                #https://stats.stackexchange.com/questions/127077/random-forest-probabilistic-prediction-vs-majority-vote
                value["output"] = np.sum(np.stack(arrs, axis=-1), axis=-1)*self.model.learning_rate 
            else:
                value["output"] = None

        #build chart data
        for key, value in feature_pairs.items():
            h,v = self._get_quantile_matrix(key[0], key[1])
            value["h_indices"] = h
            value["v_indices"] = v    

        no_predictor_features = []
        oned_features = []
        chart_data = {}
        for key, value in feature_pairs.items(): 
            newKey = key
            if value["output"] is None:
                no_predictor_features.append(key)
                value["removed"] = True
            else:          
                if collapse_1d:
                    if self._reduce_to_1d(value["output"], 0., "v"):
                        newKey = key[1]
                        value["output"] = value["output"][0,:]
                        value["h_indices"] = self.feature_ranges[newKey]
                        value["v_indices"] = None
                        value["1d_key"] = newKey
                        value["removed"] = True
                        oned_features.append(key)                 
                    elif self._reduce_to_1d(value["output"], 0., "h"):
                        newKey = key[0]
                        value["output"] = value["output"][:,0]
                        value["h_indices"] = self.feature_ranges[newKey]
                        value["v_indices"] = None
                        value["1d_key"] = newKey
                        value["removed"] = True
                        oned_features.append(key)

        #do another loop through chart_data to push 1d charts into 2d
        if collapse_1d:
            for value in list(feature_pairs.values()):
                if value["v_indices"] is None:
                    key = value["1d_key"]
                    #get list of charts with this feature
                    matchList = sorted([{"key": kInner, "feature_importance": np.std(vInner["output"])}\
                                        for kInner, vInner in feature_pairs.items()\
                                        if "removed" not in vInner and key in kInner],\
                                       key=lambda x: x["feature_importance"], reverse=True)

                    if len(matchList) > 0:
                        matchKey = matchList[0]["key"]
                        feature_pairs[matchKey]["output"] = feature_pairs[matchKey]["output"]\
                        + value["output"].reshape(\
                                                  -1 if key == matchKey[1] else 1,\
                                                  -1 if key == matchKey[0] else 1\
                                                 )

        #one last loop to generate the horizontal and vertical components
        for key, value in feature_pairs.items():
            if "removed" in value:
                pass
            else:
                value["output_H"] = np.mean(value["output"], axis=1).reshape(-1,1)
                value["output_V"] = np.mean(value["output"], axis=0).reshape(1,-1)
                value["output_HReduced"] = np.mean(value["output"] - value["output_V"].reshape(1,-1), axis=1)\
                .reshape(1,-1)
                value["output_VReduced"] = np.mean(value["output"] - value["output_H"].reshape(-1,1), axis=0)\
                .reshape(-1,1)

        #remove deleted keys
        feature_pairs = {key:val for key, val in feature_pairs.items() if "removed" not in val}
        feature_pairs = OrderedDict(sorted(feature_pairs.items(),\
                                            key=lambda x: np.std(x[1]["output"]), reverse=True))
        chart_components = {
            key: {
                "output" : val["output"],
                "output_VReduced" : val["output_VReduced"],
                "output_H" : val["output_H"],
                "output_HReduced" : val["output_HReduced"],
                "output_V" : val["output_V"]
            } for key, val in feature_pairs.items()
        }

        chart_indices = {
            key: {
                "h_indices" : val["h_indices"],
                "v_indices" : val["v_indices"]
            } for key, val in feature_pairs.items()
        }
        
        function_texts = {
            key : {
                "function_texts" : val["function_texts"]
            } for key, val in feature_pairs.items()
        } if return_text else None
        
        return chart_components, chart_indices, no_predictor_features, oned_features, function_texts

    def explain(self, fidelity_threshold = 1., rollup = None, subset_definition = None):
        """Generate an explanation for a model consisting of a subset of self.chart_components
        with r_squared greater than or equal to fidelity_threshold

            --Arguments
            fidelity_threshold: Float
                Float between 0 and 1. Stop adding chart components and return when r_squared is greater than or equal to
                this value
            rollup: String
                Not Implemented. https://github.com/MattJBritton/ForestfortheTrees/issues/7
            subset_definition: String
                If value is not None, then acts as a predicate to filter the data and generate an explanation
                for that subset. Note that this does not re-train the model or re-build chart components, but
                just generates an explanation that works best for that subset. The query should be passed
                in the format accepted by the Pandas 'where()' function, e.g. 'colA == 3 & colB == colC'
                for more info see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html
                If value is None then generate an explanation for the whole dataset
                
            --Returns
            None
                Explanation, chart components, and debugging data saved to internal state
                See _explain() for more detail.
        """           
        
        #NOTE: should be able to pass error_metric here along with fidelity_threshold
        if rollup is not None:
            raise NotImplementedError("Rollup functionality not yet implemented.")
            
        try:
            if subset_definition is not None:
                subset_indices = self.return_dataset_as_dataframe().query(subset_definition).index
                if len(subset_indices) == 0:
                    raise ValueError("Subset_definition produced empty query.")
            else:
                subset_indices = range(len(self.x))
        except:
            raise ValueError("Subset_definition is not a valid predicate.")
        
        self.explanation, self.explanation_components, self.evaluation_details\
        = self._explain(fidelity_threshold, rollup, subset_indices)    
    
    def _explain(self, fidelity_threshold = 1., rollup = None, subset_indices = None):

        """Generate an explanation for a model consisting of a subset of self.chart_components
        with r_squared greater than or equal to fidelity_threshold

            --Arguments
            fidelity_threshold: Float
                Float between 0 and 1. Stop adding chart components and return when r_squared is greater than or equal to
                this value
            rollup: String
                Not Implemented. https://github.com/MattJBritton/ForestfortheTrees/issues/7
            subset_indices: list or None
                If value is not None, then filter the set of points for which explanations are evaluated.
                Typically generated by the 'subset_definition' argument passed to 'explain()'
                If value is None then do not use and generate an explanation for the whole dataset
                
            --Returns           
            explanation: list of strings
                List of component names in the explanation, in descending order of importance
            explanation_components: dict of dict of arrays
                    Dictionary of feature pair keys with each value having output, output_v, output_h, etc
                    Each form of output is an array of prediction contributions indexed by chart_indices
            evaluation_details: list of dicts
                Each element in the list contains the details for one iteration of the greedy search algorithm.
                Each of these elements is a dictionary where the keys are feature pair names, and the correponding value
                is the r-squared when that component was selected at this position.
                Each dict also contains the score of the best feature pair, and the best feature pair itself.
                Primary use is to debug the rollup feature
        """        
        
        explanation = []   
        explanation_components = {}
        evaluation_details = [
            {
                "score": self._get_explanation_accuracy(
                    self.predictions_base,
                    "r_squared"
                )
            }
        ]
        
        while evaluation_details[-1]["score"] < fidelity_threshold\
        and len(explanation) < len(self.chart_components):
            current_details = {}
            temp_outputs = {}
            keys_to_evaluate = [key for key in self.chart_components.keys() if key not in explanation]
            for key in keys_to_evaluate:
                #roll up other keys
                current_explanation = explanation+[key]
                temp_outputs[key] = self._rollup_components(current_explanation)\
                if rollup == "advanced"\
                else self._copy_chart_components()

                current_details[key] = self._evaluate_single_explanation(
                    temp_outputs[key], 
                    current_explanation,
                    "r_squared",
                    subset_indices
                )

            #get key with highest fidelity score
            best_key = max(
                current_details.keys(),\
                key = (lambda key: current_details[key])
            )
            explanation.append(best_key)
            current_details["best_key"] = best_key

            if rollup == "simple":
                temp_outputs[best_key] = self._rollup_components(explanation)
                current_details[best_key] = self._evaluate_single_explanation(
                    temp_outputs[best_key], 
                    explanation,
                    "r_squared",
                    subset_indices
                )

            
            current_details["score"] = current_details[best_key]
            evaluation_details.append(current_details)
            explanation_components = {k : self._drop_alternate_outputs(v)\
                                      for k, v in temp_outputs[best_key].items()}
        return explanation, explanation_components, evaluation_details
        
    def cache_visualize_components(self, start = 1, end = 100, step = 1, save_to_file = False):
        
        """Cache the set of components for each model as trees are added from {start} to {end} number of trees.
        This generates the frames of a "movie" that can be played with play_components(). Caching is not required but
        speeds up playback considerably.

            --Arguments
            start: Integer
                The number of trees at which to start model playback caching
            end: Integer
                The number of trees at which to start model playback caching. Start and end define the length of the movie.
            step: Positive Integer
                How to step between {start} and {end}. If 1, view every step in the model-building process.
            save_to_file: Boolean
                If true, save the pickled components in the 'notebook_resources' folder.
                Regardless of this value, the components are always stored in the library's internal state
                
            --Returns           
            None
                Explanation, chart components, and debugging data saved to internal state
                See extract_components() for more detail.
        """        
        
        
        self.cache["play_components"] = []
        
        #prep data one time only
        dataset = self.return_dataset_as_dataframe()
        
        for i in range(start, end+1, step):
            model = self.model.clone().set_params(**{"num_estimators" : i})
            ft = ForestForTheTrees(
                dataset = dataset,
                model = model,
                task = self.task,
                target_col = self.target_col,
                sample_size = self.sample_size,
                num_tiles = self.num_tiles,
                quantiles = self.quantiles
            )
            ft.extract_components(True)
            self.cache["play_components"].append(
                {
                    "explanation" : ft.base_explanation,
                    "components" : ft.chart_components,
                    "chart_indices" : ft.chart_indices
                }
            )
            
        if save_to_file:
            for x in self.cache["play_components"]:
                for comp_key, component in x["components"].items():
                    x["components"][comp_key] = {key:val for key,val in component.items() if key == "output"}            
            with open('notebook_resources/cache_play_components.pkl', 'wb') as save_file:
                pickle.dump(self.cache["play_components"], save_file, -1)
        
    def cache_visualize_datapoints(self, save_to_file = False):
        
        """Cache the minimal and full dataset contribtions for the visualize_datapoints() method.
        This is not required but speeds rendering and interactivity.

            --Arguments
            save_to_file: Boolean
                If true, save the pickled components in the 'notebook_resources' folder.
                Regardless of this value, the components are always stored in the library's internal state
                
            --Returns           
            None
                Datapoint contributions saved to internal state
                See _get_datapoint_contributions() for more detail.
        """        
        
        minimal = self._get_datapoint_contributions(
            self.explanation_components,
            self.explanation
        )
        
        full = self._get_datapoint_contributions(
            self.explanation_components,
            self.base_explanation
        )
        
        minimal["explanation"] = "minimal"
        full["explanation"] = "full"
        
        self.cache["datapoints"] = pd.concat([minimal, full])
        
        if save_to_file:
            self.cache["datapoints"].to_csv("notebook_resources/cache_visualize_datapoints.csv")
            
    def load_cache_from_file(self):
        
        """Loads the results of prior runs of cache_visualize_components() and cache_visualization_datapoints()
        Sample verions for the bike dataset are downloaded with the library, but must be generated for
        other datasets and models.
        Note that this method only need to be called if the ForestForTheTrees object was cleared after the above methods
        were run.

            --Arguments
            None
                
            --Returns           
            None
                Loaded cache data saved to internal state
                See _cache_visualize_components() and _cache_visualize_datapoints() for more detail.
        """            
        try:
            self.cache["datapoints"] = pd.read_csv("notebook_resources/cache_visualize_datapoints.csv")
        except:
            pass
        try:
            with open('notebook_resources/cache_play_components.pkl', "rb") as input_file:
                self.cache["play_components"] = pickle.load(input_file, encoding='latin1')        
        except:
            pass
        
    def play_components(self, cache_id, auto_display = True):
        
        """Generate one frame of the "movie" showing how the explanation (output of visualize_components())
        changes as a tree is built. Frames are loaded from the cache, which is set by cache_visualize_components().

            --Arguments
            cache_id: Positive Integer
                The component to pull from the cache. Cache_id - 1 corresponds to the number of trees in the model
                at that stage of building.
            auto_display: Boolean
                If true, display the chart. Otherwise, just return it for manual alteration and display            
                
            --Returns           
            output: Altair.chart
                Returns the Altair visualization spec for further modification if necessary.
                If auto_display = True, this can be discarded.
        """ 
        
        output = self._visualize_components(
            #-1 deals with the fact that list is zero-based but number of trees starts at 1
            self.cache["play_components"][cache_id-1]["explanation"],
            self.cache["play_components"][cache_id-1]["components"],
            self.cache["play_components"][cache_id-1]["chart_indices"],
            None,
            self.cache["play_components"][cache_id-2]["components"] if cache_id > 1 else None,
            0.05,
            100,
            6
        )
        if auto_display:
            display(output)
        return output        
            
    def visualize_datapoints(self, cumulative = False, num_datapoints = 50, explanation_type = "minimal",
                            color_encoding = "prediction", auto_display = True):
        
        """Generate a line chart showing how the extracted components assemble a prediction for selected datapoints.

            --Arguments
            cumulative: Boolean
                If true, then the line chart sums predictions from each component, approaching the base model's prediction.
                Otherwise, each value of the line chart is the individual contribution of a contribution for that datapoint.
            num_datapoints: Positive Integer
                Number of datapoints to visualize (randomly selected from self.x)
            explanation_type: String
                If "minimal", then only use the components generated from the most recent run of explain().
                If "full", then use all components generated from the base model.
            color_encoding: String
                Pass the name of a feature (or "prediction" for the base model's prediction) which will be used
                as the color encoding for this chart. Lines will be colored based on this value
            auto_display: Boolean
                If true, display the chart. Otherwise, just return it for manual alteration and display.
                
            --Returns           
            output: Altair.chart
                Returns the Altair visualization spec for further modification if necessary.
                If auto_display = True, this can be discarded
        """        
        output = self._visualize_datapoints(cumulative, num_datapoints, explanation_type, color_encoding)
        if auto_display:
            display(output)
        return output
    
    def _visualize_datapoints(self, cumulative, num_datapoints, explanation_type, color_encoding):
        
        """See visualize_datapoints() above.
        """
        
        explanation_to_visualize = self.explanation\
        if len(self.explanation) > 0 and explanation_type == "minimal"\
        else self.base_explanation        
        
        if "datapoints" in self.cache and self.cache["datapoints"] is not None:
            datapoints = self.cache["datapoints"]
            datapoints = datapoints[datapoints["explanation"] == explanation_type]
        
        else:            
            datapoints = self._get_datapoint_contributions(
                self.explanation_components,
                explanation_to_visualize
            )
        
        unique_datapoint_ids = np.unique(datapoints.loc[:,"index"])
        sample_datapoint_ids = np.random.choice(unique_datapoint_ids, num_datapoints, replace = False)
        datapoints = datapoints[datapoints["index"].isin(sample_datapoint_ids)]
        datapoints = datapoints[datapoints["view"] == ("Cumulative Predictions"\
                                                       if cumulative\
                                                       else "Predictions by Chart")
                               ]
        
        df_raw = pd.DataFrame(self.x, columns = self.feature_names)
        df_raw["prediction"] = self.pred_y
        df_raw = df_raw.reset_index(drop = False)
        df_raw = df_raw[df_raw["index"].isin(sample_datapoint_ids)]       
        
        brush = alt.selection_multi()
        chart = alt.Chart(data = datapoints)\
        .mark_line()\
        .encode(
            x = alt.X(
                field = 'component',
                type = 'nominal',
                axis = alt.Axis(labelAngle = -30),
                sort = self._get_parallel_coordinate_columns(explanation_to_visualize, cumulative)
            ),
            y ='contribution:Q',
            color = alt.condition(
                brush,
                alt.Color(
                    field = color_encoding,
                    type = "quantitative",
                    scale = alt.Scale(scheme = "plasma")
                ),
                alt.value("lightgray")
            ),
            opacity = alt.condition(
                brush,
                alt.value(1.0),
                alt.value(0.2)
            ),
            tooltip = [
                alt.Tooltip(x+":"+self._get_altair_data_type(x))
                for x in self.feature_names
            ] + [
                alt.Tooltip(x+":Q")
                for x in ["prediction", "explanation_loss"]                
            ],         
            detail = 'index:N',
            order = "sort:N"
        ).transform_lookup(
            lookup = 'index',
            from_ = alt.LookupData(
                data = df_raw, 
                key = 'index',
                fields = self.feature_names
            )
        ).properties(
            height = 300,
            width = 800
        ).add_selection(
            brush
        )
        
        return chart
    
        """
        .configure_axis(
            labelColor = "white",
            tickColor = "white",
            titleColor = "white",
            grid = False
        ).configure_legend(
            labelColor = "white",
            titleColor = "white"
        )"""
        
    def visualize_components(self, plot_points = None, chart_size = 150, charts_per_row = 4):
        
        """Generate a visualization consisting of a heatmap for each self.explanation_component
        if explain() has been called, or for each self.chart_component otherwise

            --Arguments
            plot_points: Positive Integer or None
                If not None, select a random sample of this size of data points and overlay
                on each chart to indicate distribution.
            chart_size: Positive Integer
                Width and height of output chart in pixels
            charts_per_row : Positive Integer
                Number of charts to display in each row                
                
            --Returns           
            output: Altair.chart
                Returns the Altair visualization spec for further modification if necessary.
        """         
        if len(self.explanation) > 0:
            explanation_to_visualize = self.explanation
            components = self.explanation_components
        else:
            explanation_to_visualize = self.base_explanation
            components = self.chart_components
        return self._visualize_components(
            explanation_to_visualize,
            components,
            self.chart_indices,
            plot_points,
            None,
            None,
            chart_size,
            charts_per_row
        )
        
    def _visualize_components(self, explanation, components, chart_indices,
                              plot_points, ref_components, ref_components_threshold, chart_size, charts_per_row):
        
        """Generate a visualization consisting of a heatmap for the passed components in the passed explanation

            --Arguments
            explanation: List
                List of feature pair keys specifying an explanation. Typically from self.explanation or self.base_explanation
            components: dict of dict of arrays
                    Dictionary of feature pair keys with each value having output, output_v, output_h, etc
                    Each form of output is an array of prediction contributions indexed by chart_indices
            chart_indices: dict of dict of lists
                Dictionary of feature pair keys with each value having "h_indices" and "v_indices",
                each of which is a list containing the bins for the feature in the horizontal or vertical position.
                See explain() for more details.
            plot_points: Positive Integer or None
                If not None, select a random sample of this size of data points and overlay
                on each chart to indicate distribution.
            ref_components: dict of dict of lists
                Same format as components. If not None, then compare to components as display a mark in each cell that has
                changed by more than {ref_components_threshold}. Plot_points takes precedence over this if both are set.
            ref_components_threshold: Float
                Float between 0 and 1. Indicates the percentage change required for a square of the heatmap to register
                as changed when compared to {ref_components}. Default is 5% threshold = 0.05.
                If {ref_components} is None, then ignore this value.
            chart_size: Positive Integer
                Width and height of output chart in pixels
            charts_per_row : Positive Integer
                Number of charts to display in each row
                
            --Returns           
            output: Altair.chart
                Returns the Altair visualization spec for further modification if necessary.
        """         
        i = 1
        rows = []
        charts = []
        self.temp_components = components.copy()
        self.temp_indices = chart_indices.copy()
        for key in explanation:
            
            chart_df = pd.DataFrame(
                np.hstack(
                    (
                        np.array(chart_indices[key]["h_indices"]).reshape(-1,1),
                        np.array(chart_indices[key]["v_indices"]).reshape(-1,1),
                        components[key]["output"].ravel().reshape(-1,1),
                        
                        ref_components[key]["output"].ravel().reshape(-1,1)\
                        if ref_components is not None and key in ref_components\
                        else self._get_empty_sample(len(chart_indices[key]["h_indices"])).reshape(-1,1)
                    )
                ),
                columns = ["h_indices", "v_indices", "contributions", "ref_contributions"]
            )
            
            #figure out cells that should be highlighted
            chart_df["is_changed"]\
            = chart_df.apply(lambda x:
                             abs(x["ref_contributions"] - x["contributions"])\
                                 /abs(x["contributions"]+0.001) > ref_components_threshold\
                             or (x["contributions"] != 0. and key not in ref_components)#new chart this step
                             if ref_components is not None else False,
                             axis = 1
                            )

            y_encoding = alt.Y(
                field = "v_indices",
                type = "ordinal",
                sort = "descending",
                axis = alt.Axis(title = key[1])
            )                

            x_encoding = alt.X(
                field = "h_indices",
                type = "ordinal",
                sort = "ascending",
                axis = alt.Axis(
                    title = key[0],
                    labelAngle = 0,
                    labelOverlap = "greedy"
                )
            )

            color_encoding = alt.Color(
                field = "contributions",
                type = "quantitative",
                scale = alt.Scale(
                    scheme = "redblue",
                    domain = [
                        np.min([np.min(x["output"]) for x in list(self.explanation_components.values())]),
                        np.max([np.max(x["output"]) for x in list(self.explanation_components.values())])
                    ]
                ),
                legend = alt.Legend(title = "Votes")
            )
            
#             size_encoding = alt.Size(
#                 field = 
#             )
            
            tooltip_encoding = [
                alt.Tooltip('h_indices:O', title = key[0]),
                alt.Tooltip('v_indices:O', title = key[1] if key[1] != key[0] else "same"),
                alt.Tooltip("contributions:Q", title = "Contribution")
            ]

            chart = alt.Chart(data = chart_df).mark_rect()

            chart = chart.encode(
                x = x_encoding, 
                y = y_encoding, 
                color = color_encoding,
                #size = size_encoding,
                tooltip = tooltip_encoding
            ).properties(
                width = chart_size, 
                height = chart_size
            )

            if plot_points is not None:
                point_df = pd.DataFrame(self.x[np.random.choice(self.x.shape[0], plot_points, replace = False),:],\
                                  columns = self.feature_names)
                points = alt.Chart(point_df).mark_circle(
                    color = 'black',
                    size = round(chart_size/50,0)
                ).encode(
                    x = alt.X(field = key[0], type = "quantitative", sort = "ascending", axis = None),
                    y = alt.Y(field = key[1], type = "quantitative", sort = "ascending", axis = None)
                ).properties(width = chart_size, height = chart_size)
                chart = chart + points
                #chart = chart.resolve_scale(x = "independent", y = "independent")
                
            elif ref_components is not None:
                changes = alt.Chart(data = chart_df[chart_df["is_changed"]]).mark_circle(
                    size = round(chart_size/25,0)
                ).encode(
                    x = alt.X(field = "h_indices", type = "ordinal", sort = "ascending", axis = None),
                    y = alt.Y(field = "v_indices", type = "ordinal", sort = "descending", axis = None)
                ).properties(width = chart_size, height = chart_size)
                chart = chart + changes
                chart = chart.resolve_scale(y = "independent")
                
            charts.append(chart)
            if len(charts) == charts_per_row or i == len(explanation):
                rows.append(alt.hconcat(*charts))
                charts = []
            i += 1
            
        output = alt.vconcat(*rows).configure_scale(
            bandPaddingInner = 0
        )
        #dark mode
        """.configure_axis(
            labelColor = "white",
            tickColor = "white",
            titleColor = "white",
            grid = False
        ).configure_legend(
            labelColor = "white",
            titleColor = "white"
        )"""

        return output