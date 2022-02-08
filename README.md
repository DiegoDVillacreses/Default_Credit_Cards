# Credit Card Default Prediction using a function of Bank Profit as gain function

This is a notebook detailing the implementation on Python of six models to maximize a *Bank Profit* function and compare it to other standard gain and loss functions. We used information of 30,000 Taiwan's customers produced on October 2005, detailed description of the information we used can be found [on the for Machine Learning Repository of University of California, Irvine](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients). 

We use as a *Bank Profit* function: <!-- $Profit = \alpha* True Negative - (1-\alpha)*False Negative$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=Profit%20%3D%20%5Calpha*%20True%20Negative%20-%20(1-%5Calpha)*False%20Negative">, where <!-- $\alpha$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Calpha"> is a parameter that allow us to impose a relative value of a default client against a non-default. Since we doesn't know $\alpha$ we would train our models with $\alpha = (\frac{1}{3}, \frac{3}{7}, \frac{1}{2})$.

This notebook has four sections: i) data loading and handling, ii) exploratory data analysis, iii) modelling, iv) conclusions.
