"""Re-evaluate every recorded math trajectory, with no sampling or GPU use."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from agent_r1.evaluation.answers import final_answer_record
from recipes.deepscaler.math_expressions import answers_equal
from recipes.deepscaler.trajectory_reward import verify_process
from recipes.deepscaler.submission import prefilled_final_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path)
    parser.add_argument("--raw-json", type=Path, default=Path("data/raw/DeepScaleR-Preview-Dataset/deepscaler.json"))
    args = parser.parse_args()
    quality = json.loads(args.quality_report.read_text()) if args.quality_report else {}
    source_questions = {}
    for index, source_row in enumerate(json.loads(args.raw_json.read_text())):
        if isinstance(source_row, dict) and isinstance(source_row.get('problem'), str):
            source_questions.setdefault(source_row['problem'].strip(), []).append((index, str(source_row.get('answer'))))
    totals, failed = Counter(), Counter()
    failures, records, seen = {}, [], set()
    with args.input.open() as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)  # A stopped run must have complete records.
            uid = row["trajectory_uid"]
            if uid in seen: raise ValueError(f"Duplicate trajectory {uid}")
            seen.add(uid)
            last = row["steps"][-1]
            question = row["input"].removeprefix("user\n").split("\nPlease reason step by step, and put your final answer within", 1)[0]
            key = str(last.get("optimizer_weight_group_key", ""))
            index_match = re.search(r":index:(\d+)$", key)
            index = index_match[1] if index_match else None
            if index is None:
                sources = source_questions.get(question.strip(), [])
                # Warmup records have no optimizer group key. A new run can
                # already contain a independently corrected reference; trace
                # both the original and cleaned label to the same source row.
                matching = [i for i, a in sources if a == str(row['gts']) or
                            quality.get('corrections', {}).get(str(i), {}).get('answer') == str(row['gts'])]
                if not matching: raise ValueError(f"Cannot trace reference for {uid}")
                index = str(matching[0])
            gold = str(row['gts'])
            correction = quality.get("corrections", {}).get(index)
            if correction: gold = correction['answer']
            excluded = quality.get('excluded', {}).get(index)
            reasoning, submission = [], None
            for turn, step in enumerate(row['steps'], 1):
                starts_thinking = bool(re.search(r"<think>\s*$", step['input']))
                prefilled = turn == last.get('final_answer_prefill_step')
                final = (prefilled_final_record(step['output']) if prefilled else
                         final_answer_record(step['output'], require_think_close=starts_thinking))
                reasoning.append((turn, "" if prefilled else final['reasoning'] if final else step['output']))
                if final:
                    submission = final['answer']; break
            truncated_steps = last.get('truncated_policy_steps', [])
            # Old dumps contain only the cumulative token count. At a fully
            # exhausted five-turn budget every turn necessarily hit its cap.
            if 'truncated_policy_steps' not in last and (
                    last.get('generated_response_tokens') == last.get('total_response_token_budget')
                    and last.get('total_response_token_budget') and not last['has_final_answer']):
                truncated_steps = list(range(1, len(row['steps']) + 1))
            verification = verify_process(reasoning, question=question, truncated_steps=truncated_steps)
            checks = verification.audit['applicable_checks']
            process = sum(c.score for c in verification.credits)
            em = int(submission is not None and answers_equal(submission, gold))
            valid_success = int(em and bool(checks) and all(checks))
            changed = checks != last['deterministic_verification']['applicable_checks'] or abs(process-last['deterministic_process_reward']) > 1e-12
            totals.update({'trajectories': 1, 'original_correct': bool(last['terminal_em']),
                           'replayed_correct_with_clean_reference': em, 'verified_success': valid_success,
                           'checks': len(checks), 'passed': sum(checks), 'failed': len(checks)-sum(checks),
                           'no_checks': not checks, 'process_changed': changed,
                           'old_submission_rejected': bool(last['has_final_answer'] and submission is None),
                           'terminal_score_changed': em != bool(last['terminal_em']),
                           'corrected_reference_trajectories': bool(correction),
                           'quarantined_reference_trajectories': bool(excluded)})
            for segment in verification.audit['process_segment_audits']:
                for step in segment['steps']:
                    for check in step['equation_checks']:
                        if check['valid']: continue
                        eq = (check['lhs'],check['rhs'])
                        failed[eq] += 1
                        failures.setdefault(eq, {'uid':uid, 'turn':check['source_step']})
            records.append({'uid':uid,'global_step':row['global_step'],'source_index':index,
                            'reference_exclusion':excluded,'reference_corrected':bool(correction),
                            'submitted_answer':submission,'gold':gold,'terminal_em':em,
                            'verified_success':valid_success,'process_reward':process,
                            'checks':len(checks),'failed_checks':len(checks)-sum(checks),
                            'process_changed':changed,'old_terminal_em':last['terminal_em']})
            if line_number % 200 == 0: print(f"Replayed {line_number} trajectories", flush=True)
    report = {'input':str(args.input.resolve()), 'coverage':'all_complete_records', 'totals':dict(totals),
              'remaining_failed_equations':[{'lhs':a,'rhs':b,'count':n,**failures[(a,b)]} for (a,b),n in failed.most_common()],
              'records':records,
              'limitations':'Old premature submissions cannot be regenerated offline. Replayed metrics are parser diagnostics, not a new model evaluation. Unsupported contexts earn no credit.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report['totals']), flush=True)


if __name__ == '__main__':main()
