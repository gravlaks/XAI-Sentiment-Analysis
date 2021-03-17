#!/bin/bash
"""
File name: test.py

Creation Date: Wed 10 Mar 2021

Description:

"""

# Standard Python Libraries
# -----------------------------------------------------------------------------------------

# Local Application Modules
# -----------------------------------------------------------------------------------------

        """

        doc = np.array(data['tweet'][:1])
        
        
        tokenizer = Tokenizer(num_words=None,
                                   lower=True, split=' ', oov_token="UNK")

        import sys
        sys.exit()



        text_model = KerasTextClassifier(args.glove, args.prepoc,epochs=1, input_length=100)
        text_model.fit(training_data, target_data)

        pred = text_model.predict_proba(doc)
        print(pred.shape)
        
        
        
        import eli5
        from eli5.lime import TextExplainer
       
        
        te = TextExplainer(random_state=42)
        te.fit("I kill and murder. I hate you. A very violent tweet, violence damn fuck", text_model.predict_proba)
        html = te.show_prediction().data
        print(type(html))

        with open("data/data.html", "w") as file:
            file.write(html)

        """



