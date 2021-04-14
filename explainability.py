import eli5
from eli5.lime import TextExplainer


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
