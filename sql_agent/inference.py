"""Greedy Spider inference for a prepared task with prompt_ids and data_id.

Interactive tasks additionally carry database_path. Each mode takes the prompt
tokens prepared for that mode. Both return the complete trajectory.
"""
from dataclasses import asdict

from .contracts import ContractError, LengthContract, SETTINGS
from .executor import SQLiteExecutor, json_feedback_v1
from .generation import Sampling
from .graph import InteractiveSQLGraph, single_sql_fence_or_bare_v1
from .trajectory import Trajectory


async def interactive(task, generator, tokenizer, version):
    trajectory = Trajectory(task['data_id'], 'greedy', task['data_id'], version, task['prompt_ids'])
    graph = InteractiveSQLGraph(
        generator=generator, codec=tokenizer,
        executor=SQLiteExecutor(task['database_path'], timeout_seconds=30),
        length=LengthContract('total_context', 'project-v1: prompt+policy+environment <=4096'),
        sampling=Sampling(SETTINGS.eval_temperature, 1., -1, 1.), training=False,
        extract_sql=single_sql_fence_or_bare_v1, render_feedback=json_feedback_v1,
    )
    return await graph.run(trajectory)


async def direct(task, client, tokenizer, eos_ids, version):
    trace = Trajectory(task['data_id'], 'greedy', task['data_id'], version, task['prompt_ids'])
    length = LengthContract('total_context', 'project-v1: prompt+policy+environment <=4096')
    trace.length_contract = asdict(length)
    # All body fields mirror the interactive raw completion except that intermediate
    # SQL does not stop generation. No executor, feedback or second generation call.
    budget = length.remaining(len(trace.prompt_ids), 0, 0)
    response = await client.post('/v1/completions', json={
        'model':SETTINGS.model_id, 'prompt':[trace.prompt_ids], 'max_tokens':budget,
        'temperature':SETTINGS.eval_temperature, 'top_p':1., 'top_k':-1, 'repetition_penalty':1.,
        'stop':[], 'stop_token_ids':eos_ids, 'include_stop_str_in_output':True,
        'return_token_ids':True, 'add_special_tokens':False, 'stream':False, 'n':1,
    })
    response.raise_for_status()
    data = response.json()
    choice, = data['choices']
    tokens = choice['token_ids']
    text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if text != choice['text'] or len(tokens) > budget or choice['finish_reason'] not in ('length', 'stop'):
        raise ContractError('Single completion did not match its raw tokens or generation budget')
    trace.append_policy(prompt_ids=choice['prompt_token_ids'], response_ids=tokens, text=text,
                        response_id=data['id'], finish_reason=choice['finish_reason'], policy_version=version)
    trace.termination = 'sequence_limit' if choice['finish_reason']=='length' else 'model_end'
    try:
        trace.final_sql = single_sql_fence_or_bare_v1(text, 'final_sql')
    except ContractError:
        trace.termination = 'invalid_final_markup'
    trace.validate()
    return trace
