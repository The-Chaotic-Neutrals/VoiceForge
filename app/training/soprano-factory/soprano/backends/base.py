class BaseModel:
    def infer(self,
            prompts,
            top_p=0.95,
            temperature=0.3,
            repetition_penalty=1.2):
        '''
        Takes a list of prompts and returns the output hidden states
        '''
        pass

    def stream_infer(self,
            prompt,
            top_p=0.95,
            temperature=0.3,
            repetition_penalty=1.2):
        '''
        Takes a prompt and returns an iterator of the output hidden states
        '''
        pass
