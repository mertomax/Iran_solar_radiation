import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score

from sklearn.base import clone


def evaluate_data(model, data, input_columns=None, output_column=None, verbose=True):
    """
    This function train the given model with the input data, and calculate scores.

    Parameters
    ----------

    model : `Estimator, Pipeline, GridSearchCV`
    
    data : `pd.DataFrame`
        Input data.
    
    input_columns : `list`
        A list of features in input data that is fed to the model.
    
    output_column : `str`
        Target feature in the input data.

    verbose : `bool`
        Whether to print score report or not.

    return
    ------
    dict
        A dict of score , root_mean_squared_error and model.
    """

    
    if input_columns == None:
        X = data[['doysin', 'temp', 'rel_hum', 'abs_hum', 'et_rad']]
    else:
        X = data[input_columns]

    if output_column == None:
        y = data['ava_rad']
    else:
        y = data[output_column]

    if type(model) == GridSearchCV:
        model.fit(X, y)
        
        mean_cv_r2 = model.cv_results_['mean_test_score'].max()
        
        cv = KFold(n_splits=5, random_state=42, shuffle=True)
        nrmses = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv)
        nrmse = nrmses.mean()
        rmse = -nrmse

        if verbose:
            print('Mean CV R2 score      : {:.3f}'.format(mean_cv_r2))
            print('Mean CV rmse score    : {:.3f}'.format(rmse))

        return {'score': mean_cv_r2,
                'rmse': rmse,
                'model': model}
        
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
        model.fit(X_train, y_train)
    
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
    
        train_rmse = root_mean_squared_error(y_train, model.predict(X_train))
        test_rmse = root_mean_squared_error(y_test, model.predict(X_test))

        if verbose:
            print('Mean R2 score    [train : {:.3f}, test : {:.3f}]'.format(train_score, test_score))
            print('Mean rmse        [train : {:.3f}, test : {:.3f}]'.format(train_rmse, test_rmse))

        return {'score': (train_score, test_score),
                'rmse': (train_rmse, test_rmse),
                'model': model}


def evaluate_cities(model, data, **kwargs):
    """
    This function split the input data by city.
    The data of each city will be given to the evaluate_data function.
    And average and individual results of each city will be returnd.

    Parameters
    ----------

    model : `Estimator, Pipeline, GridSearchCV`
    
    data : `pd.DataFrame`
        Input data.

    return
    ------
    scores : `dict`
        R2 score of each city.
        
    rmses : `dict`
        Root mean squared error of each city.
        
    models : `dict`
        Trained models on each city.
    """

    cities = data.city.unique()
    models = {}
    scores = {}
    rmses = {}
    
    for city in cities:
        city_data = data[data.city == city]
        results = evaluate_data(clone(model), city_data, verbose=False, **kwargs)
        models[city] = results['model']
        scores[city] = results['score']
        rmses[city] = results['rmse']

    if type(model) == GridSearchCV:
        scores = pd.Series(scores)
        rmses = pd.Series(rmses)

        print('Average R2 score  : {:.3f}'.format(scores.mean()))
        print('Average rmse      : {:.3f}'.format(rmses.mean()))
        
    else:
        scores = pd.DataFrame(scores).T
        scores.rename(columns = {0:'train', 1:'test'}, inplace=True)
        rmses = pd.DataFrame(rmses).T
        rmses.rename(columns = {0:'train', 1:'test'}, inplace=True)

        print('Average R2 score                [train : {:.3f}, test : {:.3f}]'.format(scores.mean()['train'], scores.mean()['test']))
        print('Average root mean squared error [train : {:.3f}, test : {:.3f}]'.format(rmses.mean()['train'], rmses.mean()['test']))
    
    return scores, rmses, models
