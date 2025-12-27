import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from src.config import Config
from src.model import NanoHRMv3

def generate_answer(model, tokenizer, question, context, device, max_segments=8, max_new_tokens=32):
    model.eval()
    
    # Prepare Input
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=Config.MAX_SRC_LEN).to(device)
    input_ids = inputs.input_ids
    
    print(f"\nProcessing...")
    
    with torch.no_grad():
        # 1. Encode
        memory, src_mask = model.encode(input_ids)
        B, L, D = memory.shape
        
        # 2. HRM Reasoning (ACT)
        zH, zL = model.hrm_core.init_state(B, L, device)
        
        halted = False
        final_m = 0
        
        print("Thinking Steps:")
        for m in range(max_segments):
            # Forward Segment
            # During inference, we don't need to detach for gradients, but we update states
            zH, zL = model.hrm_core.forward_segment(zH, zL, memory, key_padding_mask=src_mask)
            
            # Check ACT
            q_probs = model.hrm_core.get_q_values(zH)
            q_halt = q_probs[0, 0].item()
            q_cont = q_probs[0, 1].item()
            
            print(f"  Step {m+1}: Halt Prob = {q_halt:.4f} | Cont Prob = {q_cont:.4f}")
            
            if q_halt > q_cont and m >= 1: # Basic constraint: at least 1 step ?? Or just purely ACT. 
                                         # Paper says M_min is used. Let's allow halting if prob is high.
                print(f"  -> Decided to HALT at step {m+1}")
                halted = True
                final_m = m + 1
                break
        
        if not halted:
            print(f"  -> Reached MAX steps ({max_segments})")
            final_m = max_segments

        # 3. Decode (Greedy Autoregressive)
        # Start with Pad/EOS? T5 usually uses pad_token_id as start?
        # Let's check decoder start token.
        # Ideally we use model.config.decoder_start_token_id but we don't have it set in custom config.
        # We'll use 0 (pad) as per training code assumption.
        
        decoder_input = torch.tensor([[0]], device=device) # SOS
        
        generated_tokens = []
        
        enhanced_memory = memory + zH
        
        print("Generating Answer...")
        for _ in range(max_new_tokens):
            # Re-run decoder (inefficient but simple for script)
            # Need to pass padded mask if batching, but here B=1.
            # Causal mask is handled inside model.decode if we passed it? 
            # Wait, model.decode expects 'labels' style input and generates logits for all.
            # We need to construct input incrementally.
            
            # We need to adapt model.decode or manually call layers.
            # model.decode does: dec_pos(emb) -> decoder -> head.
            # It expects `labels` argument to create `decoder_input`.
            
            # Let's manually do what's needed for generation
            tgt_emb = model.dec_pos(model.dec_embedding(decoder_input))
            tgt_len = decoder_input.size(1)
            tgt_causal_mask = torch.triu(torch.full((tgt_len, tgt_len), float('-inf'), device=device), diagonal=1)
            
            dec_out = model.decoder(
                tgt=tgt_emb,
                memory=enhanced_memory,
                tgt_mask=tgt_causal_mask,
                memory_key_padding_mask=src_mask
            )
            
            logits = model.lm_head(dec_out) # [1, Seq, Vocab]
            next_token_logits = logits[:, -1, :] # Last token
            next_token = next_token_logits.argmax(dim=-1).unsqueeze(1)
            
            token_id = next_token.item()
            if token_id == 1: # EOS for T5 is usually 1? Or tokenizer.eos_token_id?
                break
            
            generated_tokens.append(token_id)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return answer, final_m

def main():
    print("Loading Model...")
    cfg = Config()
    device = cfg.DEVICE
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER_NAME)
    model = NanoHRMv3(tokenizer, cfg).to(device)
    
    # Load Weights
    try:
        model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=device))
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Could not load weights: {e}")
        return

    print("\n=== HRM Interactive Query ===")
    print("Enter a question and context. Type 'exit' to quit.")
    
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == 'exit': break
        if not question: continue
        
        context = input("Context (press Enter for empty): ").strip()
        
        answer, steps = generate_answer(model, tokenizer, question, context, device)
        print(f"\nFinal Answer: {answer}")
        print(f"(Reasoning took {steps} steps)")

if __name__ == "__main__":
    main()
