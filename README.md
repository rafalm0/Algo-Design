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

If there is no "graphs[...].txt" on the directory, the code will generate the graphs, then export into a new "graphs[...].txt" 
file and then test it. For each of these files, the structure they follow is first there will ne a name explaining what that line
represents, (node,edge or graph) with the order of nodes then edges and the graph in the final line.

- For the line with "node" in it, it represents the id of the node, the x location, and then the y location.
- For the line with "edge" there will be the source node of that edge, the target node, and the capacity.
- For the final line, it will have a "graph" on it and it will represent: the source node by id, the target node by id, the distance
that we got from the source to the target during creating of the target, then the values of n, r and upperCap used to 
generate that graph.

From that point on the experiments will run, each method at a time, from graph to graph, in the end the code will output
a log file that can be examined for more information about the experiments results. 
(if an output log already exists it will be overwritten)

The same will then happen for experiments 2, generating at the end another log file.