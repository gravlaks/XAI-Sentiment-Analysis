import eli5
from eli5.lime import TextExplainer
import os
import webbrowser


def explain_and_save(explain_tweets_orig, explain_tweets_prep, indices, text_classifier, model_type):

    with open("data/"+model_type+"results.txt", 'w+') as f:
        f.write("Explainability metrics\n\n")
    for tweet_idx in indices:

        te = TextExplainer(random_state=42)
        te.fit(explain_tweets_prep[tweet_idx], text_classifier.predict_proba)
        metrics = te.metrics_

        with open("data/"+model_type+"results.txt", 'a') as f:

            orig_tweet = explain_tweets_orig[tweet_idx]
            f.write(orig_tweet+"\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
                print(key, value)

def get_html_single(tweet, text_classifier):
    #Show prediction for single tweet:
    te = TextExplainer(random_state=42)
    te.fit(tweet, text_classifier.predict_proba)
    html = te.show_prediction()._repr_html_()
    return html

def display_html_browser(html, name='temp'):

    path = os.path.abspath("temp/"+name+'.html')
    url = 'file://' + path


    # define the name of the directory to be created
    if not os.path.exists("temp"):
        os.mkdir("temp")

    with open(path, 'w') as f:
        f.write(html)
    webbrowser.open(url)

def save_predictions(explain_tweets_prep, indices,  text_classifier):

    if not os.path.exists("data/predictions"):
        os.mkdir("data/predictions")
    for tweet_idx in indices:
        path = f"data/predictions/html_{tweet_idx}.html"
        tweet = explain_tweets_prep[tweet_idx]
        html = get_html_single(tweet, text_classifier)
        with open(path, 'w') as f:
            f.write(html)
