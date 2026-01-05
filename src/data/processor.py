# src/data/processor.py

class PromptFormatter:
    def __init__(self, model_type="llava_vicuna"):
        """
        Args:
            model_type (str): Determines the chat template style.
                              Options: 'llava_vicuna', 'llava_mistral', 'plain'
        """
        self.model_type = model_type

    def apply_prompt(self, question, answer=None):
        """
        Wraps the raw question and answer into the specific model's format.
        
        Args:
            question (str): The input question.
            answer (str, optional): The ground truth answer (for training).
        
        Returns:
            str: The fully formatted text prompt.
        """
        
        # --- Style 1: LLaVA 1.6 Vicuna (The one in your config) ---
        if "vicuna" in self.model_type:
            # Vicuna v1.5/v1.6 standard conversation format
            # Note: We assume the image token is always first for simplicity here
            system_msg = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
            
            prompt = f"{system_msg} USER: <image>\n{question} ASSISTANT:"
            
            if answer:
                # Add EOS token </s> to tell model to stop generating
                return f"{prompt} {answer}</s>"
            else:
                return prompt

        # --- Style 2: LLaVA 1.6 Mistral / LLaMA-Adapter ---
        elif "mistral" in self.model_type or "llama" in self.model_type:
            # [INST] format
            prompt = f"[INST] <image>\n{question} [/INST]"
            
            if answer:
                return f"{prompt} {answer}</s>"
            else:
                return prompt

        # --- Fallback ---
        else:
            # Simple format
            prompt = f"Question: <image>\n{question}\nAnswer:"
            return f"{prompt} {answer}" if answer else prompt