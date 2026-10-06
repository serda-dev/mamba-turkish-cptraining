"""Small frozen generation diagnostic: task answer and JSON contract separately.

This is a regression diagnostic, not a general intelligence benchmark. Malformed
JSON whose meaning cannot be objectively extracted remains semantically unscored.
No JSON repair, retries or constrained decoding are applied.
"""
import argparse
import json
from pathlib import Path


def score_response(response, case):
    candidate = None
    decoder = json.JSONDecoder()
    for i, char in enumerate(response):
        if char == '{':
            try:
                candidate, _ = decoder.raw_decode(response[i:])
                if isinstance(candidate, dict):
                    break
            except ValueError:
                continue
    expected = case['expected']
    semantic = None if candidate is None else all(
        key in candidate and type(candidate[key]) is type(value) and candidate[key] == value
        for key, value in expected.items())
    try:
        full = json.loads(response)
        parse_valid = True
    except ValueError:
        full, parse_valid = None, False
    schema_valid = isinstance(full, dict) and set(full) == set(expected) and all(
        type(full[key]) is type(value) for key, value in expected.items())
    return {'task_correct': semantic, 'json_parse_valid': parse_valid,
            'schema_valid': schema_valid, 'contract_and_task_correct': bool(schema_valid and semantic)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--tokenizer', default=None)
    parser.add_argument('--revision', default=None)
    parser.add_argument('--cases', default='eval/contract_cases_v1.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    args = parser.parse_args()
    from src.model.load import load_model_and_tokenizer
    import torch
    cases_bytes = Path(args.cases).read_bytes()
    cases = json.loads(cases_bytes)
    model, tokenizer = load_model_and_tokenizer(args.model, args.tokenizer, revision=args.revision)
    model.eval()
    rows = []
    for case in cases:
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content':case['prompt']}],
                  tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        inputs = {key:value.to(next(model.parameters()).device) for key,value in inputs.items()}
        with torch.no_grad():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                    use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        rows.append({'id':case['id'], 'category':case['category'], 'response':response, **score_response(response,case)})
    import hashlib
    result = {'diagnostic_version':'contract_v1', 'model':args.model, 'revision':args.revision,
              'cases_sha256':hashlib.sha256(cases_bytes).hexdigest(), 'do_sample':False,
              'max_new_tokens':args.max_new_tokens, 'rows':rows,
              'summary':{key:sum(row[key] is True for row in rows) for key in
                         ('task_correct','json_parse_valid','schema_valid','contract_and_task_correct')},
              'semantic_unscored':sum(row['task_correct'] is None for row in rows), 'total':len(rows)}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result['summary']))

if __name__ == '__main__':
    main()
