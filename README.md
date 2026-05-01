# How to implement
## Simulation data
### Step 1
Run exe.py, this will generate several tables name after 'dirN/indN/dirZ/indZ + N(sample size) + .csv'. Here 'dir' stands for direct effects, 'ind' stands for indirect effects, 'Z' stands for $t = 0$, 'N' stands for $t=T$.
### Step 2
Run visualize.py, this will generate summary tables, lineplots, barplots, boxplots appearing in the paper.

## Real-world data
### Step 1
Create a file named 'test-data' under the same directory as the codes.
### Step 2 (RARD)
Download the RARD data from https://business.yelp.com/data/resources/open-dataset/.
Run realdata.py, this will generate the $X, T, Y$ under the real-data file.
### Step 3
Run exe.py, this will generate several tables name after 'dirN/indN/dirZ/indZ + N(sample size) + .csv'. Here 'dir' stands for direct effects, 'ind' stands for indirect effects, 'Z' stands for $t = 0$, 'N' stands for $t=T$.
### Step 4
Run visualize.py, this will generate summary tables, lineplots, barplots, boxplots appearing in the paper.
