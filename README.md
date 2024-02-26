# min_norm_solution_search

This is the code used for project of 2022 spring AI616.
This algorithm was used to find minimum norm solution of neural network.
The final report of this project is uploaded in this repo.

In this project, we have studied the generalization aspects of the minimum-norm interpolated solution for one-hidden-layer ReLU neural networks in teacher-student settings. Surprisingly, we have shown that if each sector given by the teacher network contains more than two training inputs, the minimum $l_2$ norm student network with zero training loss represents the same function as the teacher, i.e., the test loss is zero. Furthermore, we have empirically observed that SGD with regularization can approximate the minimum-norm solution, showing that the student network's neurons align with those of the teacher network, resulting in good generalization performance.

Our result is novel in the sense that we considered arbitrary given teacher neurons, while most works on the teacher-student setting assumed that the neurons of the teacher network are orthogonal to each other. We restricted our analysis to 2-dimensional training input, but we conjecture that our results can be extended to general input dimensions, which we leave as future work. Moreover, further analysis of how gradient methods with regularization can approximate the minimum-norm interpolated solution will be an interesting topic. We believe that our study gives a new insight into how minimizing the norm contributes to the learnability in the teacher-student setting.
