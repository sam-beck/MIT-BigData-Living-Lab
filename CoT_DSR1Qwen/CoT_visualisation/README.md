# CoT_DSR1Qwen: CoT_visualisation
A basic example of multi-sampling a LLM with multiple thought paths to achieve a CoT reasoning. 

The CoT model uses a temperature parameter below 1 to create a larger variance/distribution in output results. A sample of a set number of sequences is taken at each thought step, creating a multitude of paths in which the thought process can take. To quantify the model's certainty, an average of the model's output token's scores is taken in an attempt to determine which paths the model believes are the most likely to generate the correct result.  

The main pipeline outputs a CoT data structure as follows:
[parent_node, [child_node_1, [grandchild_node_1,[...],...]], [child_node_2, [grandchild_node_1,[...],...]]]

Where each node is defined by the following structure:
{"output":"","confidence":[]}
Where confidence refers to an array of all the generated tokens' scores / probabilities, seperated by sequence.

## [model](./model.py)
Defines the sampling parameters model and core function for generating sequences for chain of thought inference.

## [visualization](./visualization.py)
Allows for a window to be created for displaying flow charts of CoT reasoning paths. 

## [test_inference](./test_inference.py)
A basic example of a usage / testing case utilising a light-weight R1 distilled model (DSR1-Distill-Qwen-1.5B).