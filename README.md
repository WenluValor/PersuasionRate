# How to implement
## Simulation data
### Step 1
Create a file named 'test-data' under the same directory as the codes.
### Step 2
Run data.py, this will generate the $X, T, Y$ and a validation dataset under the test-data file.
### Step 3
Run exe.py, this will generate several tables name after 'dirN/indN/dirZ/indZ + N(sample size) + .csv'. Here 'dir' stands for direct effects, 'ind' stands for indirect effects, 'Z' stands for $t = 0$, 'N' stands for $t=T$. When case == 'hmm', it generates hidden markov model data. When case == 'simu', it generates correlated potential outcome data.
### Step 4
Run visualize.py, this will generate summary tables, lineplots, barplots, boxplots appearing in the paper.

## Real-world data
### Step 1
Create a file named 'real-data' under the same directory as the codes.
### Step 2 (RARD)
Download the RARD data from https://zenodo.org/records/14229945.
Run clncov.py, this will generate the $X, T, Y$ under the real-data file.
### Step 2 (CUMD)
Download the CUMD data from https://ailab.criteo.com/criteo-uplift-prediction-dataset/.
Run realdata.py, this will generate the $X, T, Y$ under the real-data file.
### Step 3
Run realexe.py, this will generate several tables name after 'dirN/indN/dirZ/indZ + N(sample size) + .csv'. Here 'dir' stands for direct effects, 'ind' stands for indirect effects, 'Z' stands for $t = 0$, 'N' stands for $t=T$.
### Step 4
Run realvisualize.py, this will generate summary tables, lineplots, barplots, boxplots appearing in the paper.
