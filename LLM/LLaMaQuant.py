from LLM.LLmMain import LLM

from langchain import LLMChain, PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp

class LLaMaQuant(LLM):
    modelPath = ""
    # nGpuLayers = 1
    # nBatch = 512
    # nCtx = 4096
    # nThreads = 2
    temperature=0.75
    top_p=1
    max_tokens=4096
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


    def __init__(self):
        self.InitializeModel()
        testStatus = self.quickTestModel()
        if testStatus == False:
            raise Exception("Model failed to pass quick test")


    def InitializeModel(self):
        self.llm = LlamaCpp(
            model_path=self.modelPath,
            callback_manager=self.callback_manager,
            verbose=True,
            # n_ctx=self.nCtx,
            # n_batch= self.nBatch,
            # n_gpu_layers=self.nGpuLayers,
            # f16_kv=True,
            temperature= self.temperature,
            max_tokens= self.max_tokens,
            verbose= True,
            top_p= self.top_p,
            streaming = True,
            seed = 0,
        )

    def quickTestModel(self):
        output = self.llm("Q: What is the capital of France? A:", stop=["Q:", "\n"],echo=False)
        if output.lower().find("paris") != -1:
            print("Test passed")
            return True
        else:
            print("Test failed")
            print(output)
            return False
        
    def respond(self, question, context):
        template = """
        [INST]
        <<SYS>>
        

        <</SYS>>
       

        Question: {question}

        [/INST]"""
        prompt = PromptTemplate(template=template, input_variables=["question", "context"])

        llm_chain = LLMChain(prompt=prompt, llm=self.llm)
        output = llm_chain({'question':question,'context':context})
        print(output)
        return output["text"]
        