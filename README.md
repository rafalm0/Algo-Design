# COMP 6651 Algorithm Design Techniques 


------------------------------------------------------------------------


## 1- Contents

* main.py
* README.md
* graphs.txt
* logs.csv

## 2 - Running the experiments

The project was coded in python 3.11 with no libraries needed to run it.

> DISCLAIMER:
> 
>   The library networkx was used ONLY to VALIDATE the results of maxFlow algorithms, this can be double checked on
>   the code if desired, it's usage was only on the "maxFlowValidator.py" script, to compare the values.


The script can be run in a simple call alongside python or python3:
```
python3 main.py
or
python main.py
```

## 3 - Running scenarios

If there is no "graphs.txt" on the directory, the code will generate the graphs, then export into a new "graphs.txt" 
file and then test it. 

From that point on the experiments will run, each method at a time, from graph to graph, in the end the code will output
a log file that can be examined for more information about the experiments results. 
(if an output log already exists it will be overwritten)

The same will then happen for experiments 2, generating at the end another log file.