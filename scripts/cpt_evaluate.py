"""Frozen heldout Turkish/English NLL and twelve generation diagnostics."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision')
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import torch
    from src.model.load import load_model_and_tokenizer
    from src.data.dataloader import ShardedMemmapPackedDataset
    from src.data.tokenize_pack import tokenization_fingerprint
    from src.cli import set_seed
    from src.eval.contracts import score_response
    set_seed(42)
    model, tokenizer = load_model_and_tokenizer(args.model, './customtokenizer',
            revision=args.revision, torch_dtype='bfloat16', device='cuda',
            attn_implementation='sdpa', use_cache=False)
    model.eval()
    dataset = ShardedMemmapPackedDataset(args.cache)
    assert dataset.manifest['cache_identity']['split'] == 'validation'
    assert dataset.manifest['tokenizer_fingerprint'] == tokenization_fingerprint(tokenizer)
    sums = {name: {'nll': 0., 'tokens': 0} for name in ('turkish', 'english')}
    device = next(model.parameters()).device
    with torch.inference_mode():
        for i in range(len(dataset)):
            item = dataset[i]
            ids = item['input_ids'].unsqueeze(0).to(device)
            mask = item['attention_mask'].unsqueeze(0).to(device)
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            losses = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                        ids[:, 1:].reshape(-1), reduction='none')
            for source_id, name in dataset.source_id_to_name.items():
                chosen = (item['token_source_ids'][1:].to(device) == source_id) & mask[0, 1:].bool()
                sums[name]['nll'] += float(losses[chosen].sum())
                sums[name]['tokens'] += int(chosen.sum())
    scores = {}
    for name, row in sums.items():
        if not row['tokens']:
            raise ValueError(f'No heldout tokens for {name}')
        nll = row['nll'] / row['tokens']
        if not math.isfinite(nll):
            raise ValueError('Nonfinite heldout NLL')
        scores[name] = {'mean_nll': nll, 'perplexity': math.exp(nll), 'tokens': row['tokens']}
    cases_path = Path('eval/contract_cases_v1.json')
    cases = json.loads(cases_path.read_text())
    diagnostics = []
    with torch.inference_mode():
        for case in cases:
            prompt = tokenizer.apply_chat_template([{'role':'user', 'content':case['prompt']}],
                    tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to(device)
            outputs = model.generate(**inputs, do_sample=False, max_new_tokens=128, use_cache=True,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            diagnostics.append({'id': case['id'], 'response': response, **score_response(response, case)})
    result = {'model':args.model, 'revision':args.revision, 'scores':scores,
              'heldout_manifest_sha256':hashlib.sha256(Path(args.cache).read_bytes()).hexdigest(),
              'diagnostic_cases_sha256':hashlib.sha256(cases_path.read_bytes()).hexdigest(),
              'diagnostics':diagnostics, 'contract_correct':sum(r['contract_and_task_correct'] for r in diagnostics)}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({'scores': scores, 'contract_correct':result['contract_correct']}), flush=True)


if __name__ == '__main__':
    main()
