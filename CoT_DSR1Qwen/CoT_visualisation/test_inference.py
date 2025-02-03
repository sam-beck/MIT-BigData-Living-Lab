import model
import visualization as canvas
import re, sys

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
llm = model.LanguageModel(model_name=model_name)

samplingParameters = model.SamplingParameters(top_p=0.15,temperature=0.1,top_k=1,best_of=1,num_beams=1,max_new_tokens=5)
results = llm.generate("If a train travels at 60 mph for 2.5 hours, how far does it travel?",samplingParameters)

# Create Application
app = canvas.QApplication(sys.argv)
view = canvas.FlowchartView()
view.setWindowTitle("CoT Visualization")
view.show()

#sentences = re.split(r'(?<=[.!?]) +', results)
print(results)
#previousNode = False
#for i in len(sentences):
#    node = view.addNode(view.convertToScreenSpace(0,i*100),sentences[i])

# Example Flowchart Nodes (Added First)
#nodeA = view.addNode(view.convertToScreenSpace(0, 0), "This is a start node with a long description that needs wrapping.")
#nodeB = view.addNode(200, 250, "Process 1")
#nodeC = view.addNode(400, 250, "Process 2")
#nodeD = view.addNode(300, 400, "End")
# Connect Nodes (Added After Nodes)
#view.addConnection(nodeA, nodeB)
#view.addConnection(nodeA, nodeC)
#view.addConnection(nodeB, nodeD)
#view.addConnection(nodeC, nodeD)

sys.exit(app.exec_())