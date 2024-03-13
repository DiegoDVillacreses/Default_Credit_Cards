# Credit Card Default Prediction using various approaches to assess class imbalance

This is a notebook detailing the implementation on Python of six models to maximize a *Bank Profit* function under heavy class imbalance and compare it to other standard gain and loss functions. We used information of 30,000 Taiwan's customers produced on October 2005, detailed description of the information we used can be found [on the for Machine Learning Repository of University of California, Irvine](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients). 

We use as a *Bank Profit* function: <!-- $Profit = \alpha* True Negative - (1-\alpha)*False Negative$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=Profit%20%3D%20%5Calpha*%20True%20Negative%20-%20(1-%5Calpha)*False%20Negative">, where <!-- $\alpha$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Calpha"> is a parameter that allow us to impose a relative value of a default client against a non-default. Since we doesn't know $\alpha$ we would train our models with <!-- $\alpha = (\frac{1}{3}, \frac{3}{7}, \frac{1}{2})$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Calpha%20%3D%20(%5Cfrac%7B1%7D%7B3%7D%2C%20%5Cfrac%7B3%7D%7B7%7D%2C%20%5Cfrac%7B1%7D%7B2%7D)">.

This notebook has four sections: i) data loading and handling, ii) exploratory data analysis, iii) modelling, iv) conclusions.

# Further research

Considering major developments in  automl, now I present examples of usage of *Bank Profit* score in autogluon (https://auto.gluon.ai/stable/index.html). Since this repository was published, many colleagues seemed the replacement of F1-Score was unnecessary. Data on how unaligned F1-Score could be with a company's profit should be included.

Also, in 2024, there is much controversy in the usage of SMOTE. Further analysis with different datasets should be done, particularly with different scores such as  *Bank Profit* score.

# Conclusions

Class Imbalance is a very common issue in the daily application of statistical methods to a broad range of problems. Here we tried to SOMTE from Nitesh et al. (2002) and a custom loss function for model selection during Hyperparameter Tunning (HPT) for LightGBM and CatBoost. We only found an improvement over no HPT (both with SMOTE) on LightGBM with our custom function for alfa = 1/3 improving from 0.1853 to 0.1991, a modest increase of 7.44%. We found no improvement on the rest of our models. Considering the importance of this area of research and the vast numerous of options to assess it we believe that more studies are needed to understand it and write more user-friendly codes.

# Bibliography

- Nitesh V Chawla, Kevin W Bowyer, Lawrence O Hall, and W Philip Kegelmeyer. (2002). Smote: synthetic minority over-sampling technique. Journal of artificial intelligence research, 16:321–357.
