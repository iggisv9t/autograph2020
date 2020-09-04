# autograph2020
My final solution for autograph2020 competition

Feedback stage: https://www.automl.ai/competitions/3  
Final stage: https://www.automl.ai/competitions/6

This solution has get 26-th place on final stage with average rank 19.6, and scored top 4 on two datasets of 5.

No neural networks used. Just old good handcrafted features and a few tricks with matirx multiplication. I also used ExtraTreesClassifier from sklearn on top of huge sparse scipy matirx, because data was too noisy.
