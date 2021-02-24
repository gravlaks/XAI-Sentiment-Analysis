from sklearn.model_selection import cross_validate, StratifiedKFold


# TODO just temp copied from Jonas' ML project.training.py
def evaluate_model(X, y, model,  gpu_mode=False):
    """
    Evaulates the model using k-fold kross validation.
    ---
    returns two dataframes: 
    1 => scores for all runs
    2 => average scores
    """
    cv = StratifiedKFold(n_splits=NUM_FOLDS, random_state=SEED, shuffle=True)
    N_JOBS = 1 if gpu_mode else -1

    scores = cross_validate(model, X, y, scoring=[
                            'f1_macro', 'f1_micro'], cv=cv, n_jobs=N_JOBS)

    averages = {}
    for key, val in scores.items():
        averages[key] = val.mean()

    return pd.DataFrame(scores), averages
