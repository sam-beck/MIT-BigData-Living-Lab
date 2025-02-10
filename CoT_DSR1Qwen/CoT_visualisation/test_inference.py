import model
import visualization as canvas
import re, sys
import math

'''
basic approach:
1) use num_return_sequences to generate multiple samples at a certain level.
2) generate using llm.generate().
3) pick a sample and add to end of the prompt.
4) repeat 2 and 3 until convergence or max generations completed...
'''

# Basic form to display the CoT data to a tree of nodes (with decoded / stringified node outputs) in flow chart form. Also outputs the average confidence level for each sequence 
def createNodes(x,y,arr,width=100,shiftAmt=500,dropAmt=250,shiftReduction=2,previousNode=None):
    if not isinstance(arr[0],str):
        if "confidence" in arr[0].keys():
            # Rounds to 2 DP
            info = ("\n\n Confidence: " + str(round(sum(arr[0]["confidence"])/len(arr[0]["confidence"]),2)))
            # Add current node to tree
            currentNode = view.addNode(x,y,arr[0]["output"] + info,width)
        else:
            # Add current node to tree
            currentNode = view.addNode(x,y,arr[0]["output"],width)
    else:
        # Add current node to tree
        currentNode = view.addNode(x,y,arr[0],width)
    # Add connections to tree
    if previousNode is not None:
        view.addConnection(previousNode,currentNode)
    # Shift amount, depends on #nodes
    if (len(arr)-1) % 2 == 1:
        shift = -math.floor((len(arr)-1)/2) * shiftAmt
    else:
        shift = -(len(arr)-1)/2 * (shiftAmt/2)
    # Recursive visual node generation
    for i in range(1,len(arr)):
        createNodes(x+shift,y+(dropAmt/2 if previousNode is None else dropAmt),arr[i],width,shiftAmt/shiftReduction,dropAmt,shiftReduction,currentNode)
        shift += shiftAmt

# Load model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Input prompt
input_text = "If a train travels at 60 mph for 2.5 hours, how far does it travel?"

# Number of nodes that are created at each node
nodes = 2
# Depth of tree
length = 2

# Get model
llm = model.LanguageModel(model_name=model_name)

# Generates 75 tokens at each node
samplingParameters = model.SamplingParameters(temperature=0.8,top_p=0.7,max_new_tokens=75,num_return_sequences=nodes)

# Generates CoT tree using decoded string outputs
output = llm.CoTTreeStrings(samplingParameters,input_text,length)

# Get output
print("Output CoT Tree:\n")
print(output)

# Create Application
app = canvas.QApplication(sys.argv)
view = canvas.FlowchartView()
view.setWindowTitle("CoT Visualization")
view.show()

# Display nodes
createNodes(view.windowWidth/2,0,output,width=350,shiftAmt=600,dropAmt=350,shiftReduction=1.6)

sys.exit(app.exec_())