"""New CVLT common-history paired evaluation; no old output is overwritten."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def now():
    return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def helpers():
    path = ROOT / 'scripts/analysis/cvlt_control_draft_20261007/prepare.py'
    spec = importlib.util.spec_from_file_location('cvlt_archive_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser, runtime = module.frozen_helpers()
    return parser, runtime['SYSTEM_PROMPT'], module.score_response


class Client:
    def __init__(self, model, spec, output):
        self.model, self.spec, self.output = model, spec, output
        self.url = 'http://' + os.environ['OLLAMA_HOST']
        self.latencies = []
        self.actual_calls = 0
        self.serving_checked = False

    def serving(self):
        table = subprocess.check_output(['ollama', 'ps'], text=True)
        line = next((l for l in table.splitlines() if l.split() and l.split()[0] == self.model), None)
        if line is None or '100% GPU' not in line:
            raise RuntimeError(f'Full GPU serving not established: {table}')
        match = re.search(r'100% GPU\s+(\d+)', line)
        if not match or int(match.group(1)) != self.spec['inference']['context_tokens']:
            raise RuntimeError(f'Unexpected served context: {table}')
        native = json.loads(urllib.request.urlopen(self.url + '/api/ps', timeout=20).read())
        log_path = self.output.parent/f"ollama_{os.environ.get('SLURM_JOB_ID')}.log"
        load_lines = [line for line in log_path.read_text().splitlines()
                      if 'load request' in line and 'Operation:commit' in line] if log_path.exists() else []
        load_line = load_lines[-1] if load_lines else None
        threads_match = re.search(r'NumThreads:\s*(\d+)',load_line) if load_line else None
        actual_threads = int(threads_match.group(1)) if threads_match else None
        write(self.output / 'serving.json', {'captured_at': now(), 'ollama_ps': table,
             'native_ps': native, 'hostname': socket.gethostname(),
             'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total',
                    '--format=csv,noheader'], text=True),
             'ollama_version': subprocess.run(['ollama','--version'],text=True,capture_output=True).stdout,
             'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
             'server_log':str(log_path),'committed_load_request_line':load_line,
             'actual_server_num_threads':actual_threads})
        if self.spec['inference']['endpoint'].startswith('native Ollama /api/chat') and actual_threads != self.spec['inference']['server_threads']:
            raise RuntimeError(f"Actual server NumThreads={actual_threads}, expected {self.spec['inference']['server_threads']}; see {log_path}")
        self.serving_checked = True

    def call(self, payload, identity):
        received = self.output/'received'/f'{identity}.json'
        attempt_dir = self.output/'received_attempts'/identity
        prior_attempts = sorted(attempt_dir.glob('attempt_*.json'))
        if received.exists() or prior_attempts:
            preserved = received if received.exists() else prior_attempts[0]
            raise RuntimeError('A prior response or attempt is preserved but not committed to calls; stop without another inference or overwrite: ' + str(preserved))
        attempts = []
        native = self.spec['inference']['endpoint'].startswith('native Ollama /api/chat')
        wire_request = {'model':payload['model'],'messages':payload['messages'],'stream':False,'think':False,
                        'options':{'temperature':payload['temperature'],'num_predict':payload['max_tokens'],
                                   'top_p':1.0,'num_ctx':self.spec['inference']['context_tokens'],
                                   'num_thread':self.spec['inference']['server_threads']}} if native else payload
        request_data = json.dumps(wire_request, ensure_ascii=False).encode()
        endpoint = '/api/chat' if native else '/v1/chat/completions'
        max_attempts = self.spec['inference']['max_transport_attempts']

        def record_attempt(record, response):
            attempts.append(record)
            write(attempt_dir/f"attempt_{record['attempt']}.json",
                  {'request':payload,'wire_request':wire_request,'raw_response':response,'attempt':record})
            with (self.output/'attempts.jsonl').open('a') as handle:
                handle.write(json.dumps({'identity':identity,**record})+'\n')

        def preserve_final(response, status, error=None):
            value = {'request':payload,'wire_request':wire_request,'raw_response':response,'attempts':attempts,
                     'response_status':status,'selected_attempt':attempts[-1]['attempt'] if status=='selected_valid_response' else None,
                     'final_attempt':attempts[-1]['attempt']}
            if error is not None:
                value['error'] = error
            write(received,value)

        for attempt in range(max_attempts):
            start = time.perf_counter()
            request = urllib.request.Request(self.url + endpoint,
                data=request_data,
                headers={'Content-Type':'application/json','Authorization':'Bearer ollama'})
            self.actual_calls += 1
            try:
                response = json.loads(urllib.request.urlopen(request, timeout=300).read())
            except (urllib.error.URLError, TimeoutError) as error:
                record_attempt({'attempt':attempt+1,'status':'transport_error','elapsed_seconds':time.perf_counter()-start,
                                'error':str(error)},None)
                if isinstance(error, urllib.error.HTTPError) and error.code not in (429,500,502,503,504):
                    preserve_final(None,'final_invalid_response',str(error))
                    raise
                if attempt+1 == max_attempts:
                    preserve_final(None,'final_invalid_response',str(error))
                    raise
                time.sleep(2**attempt)
                continue
            elapsed = time.perf_counter() - start
            try:
                choice = None if native else response['choices'][0]
                message = response['message'] if native else choice['message']
                if message.get('reasoning') or message.get('reasoning_content') or message.get('thinking'):
                    raise RuntimeError('Unexpected reasoning channel with reasoning_effort=none')
                content = message.get('content','')
                if not isinstance(content,str):
                    raise RuntimeError('Response content is not text')
                finish = response.get('done_reason') if native else choice.get('finish_reason')
                if finish not in ('stop','length',None):
                    raise RuntimeError('Unsupported finish reason')
                missing = []
                if native and response.get('done') is not True:
                    missing.append('done')
                if finish is None:
                    missing.append('done_reason' if native else 'finish_reason')
                usage = {'prompt_tokens':response.get('prompt_eval_count'),
                         'completion_tokens':response.get('eval_count')} if native else response.get('usage')
                prompt_tokens = completion_tokens = None
                if not usage:
                    missing.append('usage')
                else:
                    if usage.get('prompt_tokens') is None or int(usage['prompt_tokens']) <= 0:
                        missing.append('prompt_eval_count' if native else 'prompt_tokens')
                    else:
                        prompt_tokens = int(usage['prompt_tokens'])
                    if usage.get('completion_tokens') is None:
                        missing.append('eval_count' if native else 'completion_tokens')
                    else:
                        completion_tokens = int(usage['completion_tokens'])
                        if completion_tokens < 0:
                            raise RuntimeError('Reported completion tokens are negative')
                # num_ctx bounds the live cache window. The default runner can
                # shift that window while generating, so cumulative input plus
                # output is not an API validity bound.
                if prompt_tokens is not None and prompt_tokens > self.spec['inference']['context_tokens']:
                    raise RuntimeError('Reported input tokens exceed served context')
            except (KeyError,IndexError,AttributeError,TypeError,ValueError,RuntimeError) as error:
                record_attempt({'attempt':attempt+1,'status':'hard_response_error','elapsed_seconds':elapsed,
                                'error':str(error)},response)
                preserve_final(response,'final_invalid_response',str(error))
                raise
            if missing:
                error = 'Missing terminal response metadata: ' + ', '.join(missing)
                record_attempt({'attempt':attempt+1,'status':'missing_terminal_metadata','elapsed_seconds':elapsed,
                                'missing_metadata':missing,'error':error},response)
                if attempt+1 == max_attempts:
                    preserve_final(response,'final_invalid_response',error)
                    raise RuntimeError(error + f' after {max_attempts} transport attempts')
                time.sleep(2**attempt)
                continue
            record_attempt({'attempt':attempt+1,'status':'valid_response','elapsed_seconds':elapsed},response)
            preserve_final(response,'selected_valid_response')
            if native:
                usage['total_tokens'] = prompt_tokens + completion_tokens
            self.latencies.append(elapsed)
            if not self.serving_checked:
                self.serving()
            return {'response':content.strip(),'raw_content':content,'finish_reason':finish,
                    'usage':usage,'latency_seconds':elapsed,'attempts':attempts,
                    'protocol_valid':finish=='stop','raw_response':response,'wire_request':wire_request,
                    'prompt_plus_completion_exceeds_context':prompt_tokens+completion_tokens > self.spec['inference']['context_tokens'],
                    'generation_shift_trace_recorded':False}


def run_one(model, spec, items, output, client_factory=Client, offline=False):
    from .token_matching import Counter, Matcher
    parser, system, score = helpers()
    output.mkdir(parents=True, exist_ok=True)
    copy = output/'SPEC.json'
    if copy.exists() and json.loads(copy.read_text()) != spec:
        raise RuntimeError('Existing output belongs to a different specification')
    write(copy,spec)
    start=time.perf_counter();records=[];baseline={};histories={}
    client=client_factory(model,spec,output)
    with Counter(model) as counter:
        metadata_path=output/'tokenizer.json'
        if metadata_path.exists() and json.loads(metadata_path.read_text()) != counter.metadata:
            raise RuntimeError('Local model/tokenizer metadata changed; do not mix completed calls')
        write(metadata_path,counter.metadata)
        def request(item, trial_index, condition, matcher, replies):
            user, matching, replacements = matcher.build(trial_index,replies,condition)
            if not matching['matched']:
                raise RuntimeError('Context token matching failed')
            raw_budget=counter.count(user)+counter.count(system)
            if raw_budget+128 > spec['inference']['context_tokens']:
                write(output/'context_budget_failure.json',{'task_id':item['task_id'],'trial':trial_index+1,
                      'condition':condition,'raw_tokens':raw_budget,'prompt':user})
                raise RuntimeError('Raw prompt approaches served context; no silent history truncation')
            payload={'model':model,'messages':[{'role':'system','content':system},{'role':'user','content':user}],
                     'temperature':spec['inference']['temperature'],'max_tokens':spec['inference']['max_tokens'],
                     'reasoning_effort':spec['inference']['reasoning_effort']}
            identity=f"{item['task_id']}_{trial_index+1}_{condition}"
            path=output/'calls'/f'{identity}.json'
            if path.exists():
                record=json.loads(path.read_text())
                if record['request'] != payload:
                    raise RuntimeError('Completed call no longer matches same request; cannot reuse')
            else:
                reply=client.call(payload,identity)
                turn=item['parameters']['turns'][trial_index]
                record={'model':model,'family':next(m['family'] for m in spec['model_panel'] if m['model']==model),
                        'task_id':item['task_id'],'difficulty':item['difficulty'],'trial':trial_index+1,
                        'turn_type':turn['type'],'condition':condition,'request':payload,'token_matching':matching,
                        'replacements':replacements,'gold_expected_words':turn['expected_words'],
                        'synthetic_history':offline,'resource_pilot':len(items)!=50,
                        'captured_at':now(),**reply}
                record['score']=score(reply['response'],turn['expected_words'],item['parameters']['primary_list'],
                                      item['parameters']['interference_list'],parser)
                write(path,record)
            records.append(record)
            return record
        # All new C0 episodes are collected before any branch. Only C0 replies
        # enter the common histories, with archived outer-whitespace stripping.
        for item in items:
            replies=[];matcher=Matcher(counter,item)
            for index,turn in enumerate(item['parameters']['turns']):
                record=request(item,index,'C0',matcher,replies)
                baseline[(item['task_id'],index)]=record
                replies.append(record['response'])
            histories[item['task_id']]=replies
            write(output/'progress.json',{'phase':'C0','completed_calls':len(records),'last_task':item['task_id']})
            print(f"{model} C0 task {item['task_id']} complete; calls={len(records)}",flush=True)
        model_index=next(i for i,m in enumerate(spec['model_panel']) if m['model']==model)
        probe_index=0
        for item in items:
            matcher=Matcher(counter,item);replies=histories[item['task_id']]
            for index,turn in enumerate(item['parameters']['turns']):
                if turn['type'] not in ('short_delay_recall','long_delay_recall'):
                    continue
                conditions=['C1','C2','C3'];rotation=(probe_index+model_index)%3
                conditions=conditions[rotation:]+conditions[:rotation]
                for condition in conditions:
                    record=request(item,index,condition,matcher,replies)
                    original=baseline[(item['task_id'],index)]
                    delta=record['usage']['prompt_tokens']-original['usage']['prompt_tokens']
                    record['server_prompt_tokens_delta_vs_C0']=delta
                    write(output/'calls'/f"{item['task_id']}_{index+1}_{condition}.json",record)
                    if delta != 0:
                        write(output/'server_token_mismatch.json',{'C0':original,'branch':record})
                        raise RuntimeError('Actual server prompt-token counts differ despite local raw match')
                probe_index+=1
            write(output/'progress.json',{'phase':'branches','completed_calls':len(records),'last_task':item['task_id']})
    expected=sum(len(i['parameters']['turns']) for i in items)+3*sum(
        t['type'] in ('short_delay_recall','long_delay_recall') for i in items for t in i['parameters']['turns'])
    if len(records)!=expected:
        raise RuntimeError('Incomplete authorized call count')
    summary={'status':'offline_scheduler_check_complete' if offline else ('resource_pilot_complete' if len(items)!=50 else 'formal_model_complete'),'model':model,
             'synthetic_only':offline,'model_inference_calls_this_invocation':0 if offline else client.actual_calls,
             'started_at':datetime.fromtimestamp(time.time()-(time.perf_counter()-start),ZoneInfo('Asia/Tokyo')).isoformat(),
             'completed_at':now(),'logical_calls':len(records),'actual_calls_this_invocation':client.actual_calls,
             'expected_logical_calls':expected,'wall_seconds':time.perf_counter()-start,'latencies_this_invocation':client.latencies,
             'all_actual_probe_prompt_counts_equal_to_C0':True,
             'blank_responses':sum(not r['response'] for r in records),
             'length_finish_responses':sum(r['finish_reason']=='length' for r in records),
             'resource_pilot':len(items)!=50,'pilot_effects_not_reported':len(items)!=50,
             'conditions':['C0','C1','C2','C3'],'repetitions':1,'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
    write(output/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['pilot','formal'],required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--model')
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Inference only inside a Slurm GPU allocation')
    spec=json.loads((HERE/'SPEC.json').read_text());items=[json.loads(l) for l in Path(spec['materials']).read_text().splitlines()]
    if len(items)!=50 or sum(len(i['parameters']['turns']) for i in items)!=338:
        raise RuntimeError('Authorized material coverage changed')
    selected=args.model or spec['pilot']['model']
    if selected not in [m['model'] for m in spec['model_panel']]:
        raise RuntimeError('Model not in authorized twelve-model panel')
    if args.mode=='pilot':
        items=[i for i in items if i['task_id'] in spec['pilot']['task_ids']]
    result=run_one(selected,spec,items,args.output/selected.replace(':','__'))
    print(json.dumps({k:result[k] for k in ['status','model','logical_calls','wall_seconds']}),flush=True)


if __name__=='__main__':
    main()
