class BaseModel:    

    # def __init__(self, **kwargs):
    #     """Initialize the model.

    #     Parameters
    #     ----------
    #     **kwargs
    #         Parameters of the model.
    #     """
    #     self.set_params(**kwargs)
    def __init__(self):
        self._model = None
    

    def fit(self, X, y):
        """Fit the model to the data.

        Parameters
        ----------
        X : array-like
            Input data.
        y : array-like
            Output data.

        Returns
        -------
        self
            Fitted model.
        """
        return self._model.fit(X, y)

    def predict(self, X):
        """Predict the output of the model.

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        array-like
            Predicted output.
        """
        # raise NotImplementedError
        return self._model.predict(X)
    
    def predict_proba(self, X):
        """Get the inclusion probability for each sample.

        Arguments
        ---------
        X: numpy.ndarray
            Feature matrix to predict.

        Returns
        -------
        numpy.ndarray
            Array with the probabilities for each class, with two
            columns (class 0, and class 1) and the number of samples rows.
        """
        return self._model.predict_proba(X)

    def score(self, X, y):
        """Get the score of the model.

        Parameters
        ----------
        X : array-like
            Input data.
        y : array-like
            Output data.

        Returns
        -------
        float
            Score of the model.
        """
        raise NotImplementedError

    def save(self, path):
        """Save the model to a file.

        Parameters
        ----------
        path : str
            Path to the file.
        """
        raise NotImplementedError

    def load(self, path):
        """Load the model from a file.

        Parameters
        ----------
        path : str
            Path to the file.
        """
        raise NotImplementedError

    def get_params(self, deep=True):
        """Get the parameters of the model.

        Parameters
        ----------
        deep : bool, optional
            Get the parameters of the model, by default True

        Returns
        -------
        dict
            Parameters of the model.
        """
        return self.param

    def set_params(self, **parameters):
        """Set the parameters of the model.

        Parameters
        ----------
        **parameters
            Parameters of the model.

        Returns
        -------
        self
            Model with new parameters.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

#create base model class for all models 

#         """
#         return self._param
#
#     @param.setter
#     def param(self, value):
#         """Set the parameters of the model.
#
#         Parameters
#         ----------
#         value : dict
#             Dictionary with parameter: value.
#         """
#         for par in value:
#             setattr(self, par, value[par])
#
#     def __repr__(self):
#         """Get the representation of the model.
#
#         Returns
#         -------
#         str
#             Representation of the model.
#         """
#         return f"{self.name}({self.param})"
#
#     def __str__(self):
#         """Get the string representation of the model.
#
#         Returns
#         -------
#         str
#             String representation of the model.
#         """
#         return self.__repr__()
#
#     def __eq__(self, other):
#         """Check if two models are equal.
#
#         Parameters
#         ----------
#         other : object
#             Other object to compare with.
#
#         Returns
#         -------
#         bool
#             True if equal, False otherwise.
#         """
#         if not isinstance(other, BaseModel):
#             return False
#         return self.param == other.param
#
#     def __hash__(self):
#         """Get the hash of the model.
#
#         Returns
#         -------
#         int
#             Hash of the model.
#         """
#         return hash(self.__repr__())
#
#     def fit(self, X, y):
#         """Fit the model to the data.
#
#         Parameters
#         ----------
#         X : array-like
#             Input data.
#         y : array-like
#             Output data.
#
#         Returns
#         -------
#         self
#             Fitted model.
#         """
#         return self
#
#     def predict(self, X):
#         """Predict the output of the model.
#
#         Parameters
#         ----------
#         X : array-like
#             Input data.
#
#         Returns
#         -------
#         array-like
#             Predicted output.
#         """
#         raise NotImplementedError
#
#     def score(self, X, y):
#         """Get the score of the model.
#
#         Parameters
#         ----------
#         X : array-like
#             Input data.
#         y : array-like
#             Output data.
#
#         Returns
#         -------
#         float
#             Score of the model.
#         """
#         raise NotImplementedError
#
#     def save(self, path):
#         """Save the model to a file.
#
#         Parameters
#         ----------
#         path : str
#             Path to the file.
#         """
#         raise NotImplementedError
#
#     def load(self, path):
#         """Load the model from a file.
#
#         Parameters
#         ----------
#         path : str
#             Path to the file.
#         """
#         raise NotImplementedError
#
#     def get_params(self, deep=True):
#         """Get the parameters of the model.
#
#         Parameters
#         ----------
#         deep : bool, optional
#             Get the parameters of the model, by default True
#
#         Returns
#         -------
#         dict
#             Parameters of the model.
#         """
#         return self.param
#
#     def set_params(self, **parameters):
#         """Set the parameters of the model.
#
#         Parameters
#         ----------
#         **parameters
#             Parameters of the model.
#
#         Returns
#         -------
#         self
#             Model with new parameters.
#         """
#         for parameter, value in parameters.items():
#             setattr(self, parameter, value)
#         return self
#
#     def get_param_names(self):
#         """Get the names of the parameters of the model.
#
#         Returns
#         -------
#         list
#             Names of the parameters of the model.
#         """
#         return list(self.param.keys())
#
#     def get_param_values(self):

#         """Get the values of the parameters of the model.
#
#         Returns
#         -------
#         list
#             Values of the parameters of the model.
#         """
#         return list(self.param.values())
#
#     def get_param_items(self):
#         """Get the items of the parameters of the model.
#
#         Returns
#         -------
#         list
#             Items of the parameters of the model.
#         """
#         return list(self.param.items())
#
#     def get_param_types(self):
#         """Get the types of the parameters of the model.
#
#         Returns
#         -------
#         list
#             Types of the parameters of the model.
#         """
#         return [type(x) for x in self.param.values()]
#
#     def get_param_by_name(self, name):
#         """Get the value of a parameter of the model.
#
#         Parameters
#         ----------
#         name : str
#             Name of the parameter.
#
#         Returns
#         -------
#         object
#             Value of the parameter.
#         """
#         return self.param[name]
#
#     def get_param_by_index(self, index):
#         """Get the value of a parameter of the model.
#
#         Parameters
#         ----------
#         index : int
#             Index of the parameter.
#
#         Returns
#         -------
#         object
#             Value of the parameter.
#         """
#         return self.param[self.get_param_names()[index]]
#
#     def set_param_by_name(self, name, value):
#         """Set the value of a parameter of the model.
#
#         Parameters
#         ----------
#         name : str
#             Name of the parameter.
#         value : object
#             Value of the parameter.
#
#         Returns
#         -------
#         self
#             Model with new parameter value.
#         """
#         self.param[name] = value
#         return self
#
#     def set_param_by_index(self, index, value):
#         """Set the value of a parameter of the model.
#
#         Parameters
#         ----------
#         index : int
#             Index of the parameter.
#         value : object
#             Value of the parameter.
#
#         Returns
#         -------
#         self
#             Model with new parameter value.
#         """
#         self.param[self.get_param_names()[index]] = value
#         return self
#
#     def set_params_by_name(self, **parameters):
#         """Set the values of the parameters of the model.
#
#         Parameters
#         ----------
#         **parameters
#             Parameters of the model.
#
#         Returns
#         -------
#         self
#             Model with new parameters.
#         """
#         for parameter, value in parameters.items():
#             self.param[parameter] = value

#         return self
#
#     def set_params_by_index(self, **parameters):
#         """Set the values of the parameters of the model.
#
#         Parameters
#         ----------
#         **parameters
#             Parameters of the model.
#
#         Returns
#         -------
#         self
#             Model with new parameters.
#         """
#         for parameter, value in parameters.items():
#             self.param[self.get_param_names()[parameter]] = value
#         return self
#
#     def get_param_index(self, name):
#         """Get the index of a parameter of the model.
#
#         Parameters
#         ----------
#         name : str
#             Name of the parameter.
#
#         Returns
#         -------
#         int
#             Index of the parameter.
#         """
#         return self.get_param_names().index(name)
#
#     def get_param_name(self, index):
#         """Get the name of a parameter of the model.
#
#         Parameters
#         ----------
#         index : int
#             Index of the parameter.
#
#         Returns
#         -------
#         str
#             Name of the parameter.
#         """
#         return self.get_param_names()[index]
#
#     def get_param_type(self, name):
#         """Get the type of a parameter of the model.
#
#         Parameters
#         ----------
#         name : str
#             Name of the parameter.
#
#         Returns
#         -------
#         type
#             Type of the parameter.
#         """
#         return type(self.param[name])
#
#     def get_param_type_by_index(self, index):

#         """Get the type of a parameter of the model.
#
#         Parameters
#         ----------
#         index : int
#             Index of the parameter.
#
#         Returns
#         -------
#         type
#             Type of the parameter.
#         """
#         return type(self.param[self.get_param_names()[index]])
#
#     def get_param_type_by_name(self, name):
#         """Get the type of a parameter of the model.
#
#         Parameters
#         ----------
#         name : str
#             Name of the parameter.
#
#         Returns
#         -------
#         type
#             Type of the parameter.
#         """
#         return type(self.param[name])

#     def get_param_default(self, name):
#         """Get the default value of a parameter of the model.
#
#         Parameters
#         ----------
#         name : str
#             Name of the parameter.
#
#         Returns
#         -------
#         object
#             Default value of the parameter.
#         """
#         return self.default_param[name]
#
#     def get_param_default_by_index(self, index):
#         """Get the default value of a parameter of the model.
#
#         Parameters
#         ----------
#         index : int
#             Index of the parameter.
#
#         Returns
#         -------
#         object
#             Default value of the parameter.
#         """
#         return self.default_param[self.get_param_names()[index]]
#
#     def get_param_default_by_name(self, name):
#         """Get the default value of a parameter of the model.
#
#         Parameters
#         ----------
#         name : str
#             Name of the parameter.
#
#         Returns
#         -------
#         object
#             Default value of the parameter.
#         """
#         return self.default_param[name]
#
#     def get_param_default_type(self, name):
#         """Get the default type of a parameter of the model.
#
#         Parameters
#         ----------
#         name : str
#             Name of the parameter.
#
#         Returns
#         -------
#         type
#             Default type of the parameter.
#         """
#         return type(self.default_param[name])
#
#     def get_param_default_type_by_index(self, index):
#         """Get the default type of a parameter of the model.
#
#         Parameters
#         ----------
#         index : int
#             Index of the parameter.
#
#         Returns
#         -------
#         type
#             Default type of the parameter.
#         """
#         return type(self.default_param[self.get_param_names()[index]])

#     def get_param_default_type_by_name(self, name):
#         """Get the default type of a parameter of the model.
#
#         Parameters

#         ----------
#         name : str
#             Name of the parameter.
#
#         Returns
#         -------
#         type
#             Default type of the parameter.
#         """
#         return type(self.default_param[name])
#
#     def get_param_default_value(self, name):
#         """Get the default value of a parameter of the model.
#
#         Parameters
#         ----------
#         name : str
#             Name of the parameter.
#
#         Returns
#         -------
#         object
#             Default value of the parameter.
#         """
#         return self.default_param[name]
#
#     def get_param_default_value_by_index(self, index):
#         """Get the default value of a parameter of the model.
#
#         Parameters
#         ----------
#         index : int
#             Index of the parameter.
#
#         Returns
#         -------
#         object
#             Default value of the parameter.
#         """
#         return self.default_param[self.get_param_names()[index]]
#
#     def get_param_default_value_by_name(self, name):
#         """Get the default value of a parameter of the model.
#
#         Parameters
#         ----------
#         name : str
#             Name of the parameter.
#
#         Returns
#         -------
#         object
#             Default value of the parameter.
#         """
#         return self.default_param[name]
#
#     def get_param_default_type_value(self, name):
#         """Get the default type and value of a parameter of the model.
#
#         Parameters
#         ----------
#         name : str
#             Name of the parameter.
#
#         Returns
#         -------
#         tuple
#             Default type and value of the parameter.
#         """
#         return type(self.default_param[name]), self.default_param[name]
#
#     def get_param_default_type_value_by_index(self, index):
#         """Get the default type and value of a parameter of the model.
#
#         Parameters




