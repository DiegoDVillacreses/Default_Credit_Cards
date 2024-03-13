# Credit Card Default Prediction using various approaches to assess class imbalance

This is a notebook detailing the implementation on Python of six models to maximize a *Bank Profit* function under heavy class imbalance and compare it to other standard gain and loss functions. We used information of 30,000 Taiwan's customers produced on October 2005, detailed description of the information we used can be found [on the for Machine Learning Repository of University of California, Irvine](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients). 

We use as a *Bank Profit* function:  $Average Profit_{per costumer} = \\frac{ \\sum_i(\\alpha* True Negative_i - (1-\\alpha)*False Negative_i)}{Total True Negative + Total False Negative)}$, where $\alpha$ is a parameter that allow us to impose a relative value for a client default against a non-default. Since we don't know $\alpha$ we train our models with $\alpha = (\frac{1}{3}, \frac{3}{7}, \frac{1}{2})$.

This notebook has four sections: i) data loading and handling, ii) exploratory data analysis, iii) modelling, iv) conclusions.

# Further research

Considering major developments in  automl, now I present examples of usage of *Bank Profit* score in autogluon (https://auto.gluon.ai/stable/index.html). Since this repository was published, many colleagues seemed the replacement of F1-Score was unnecessary. Data on how unaligned F1-Score could be with a company's profit should be included.

Also, in 2024, there is much controversy in the usage of SMOTE. Further analysis with different datasets should be done, particularly with different scores such as  *Bank Profit* score.

# Conclusions

Class Imbalance is a very common issue in the daily application of statistical methods to a broad range of problems. Here we tried to SOMTE from Nitesh et al. (2002) and a custom loss function for model selection during Hyperparameter Tunning (HPT) for LightGBM and CatBoost. We only found an improvement over no HPT (both with SMOTE) on LightGBM with our custom function for alfa = 1/3 improving from 0.1853 to 0.1991, a modest increase of 7.44%. We found no improvement on the rest of our models. Considering the importance of this area of research and the vast numerous of options to assess it we believe that more studies are needed to understand it and write more user-friendly codes.

# Bibliography

- Nitesh V Chawla, Kevin W Bowyer, Lawrence O Hall, and W Philip Kegelmeyer. (2002). Smote: synthetic minority over-sampling technique. Journal of artificial intelligence research, 16:321–357.
